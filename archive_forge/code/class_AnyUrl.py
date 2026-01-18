import re
from ipaddress import (
from typing import (
from . import errors
from .utils import Representation, update_not_none
from .validators import constr_length_validator, str_validator
class AnyUrl(str):
    strip_whitespace = True
    min_length = 1
    max_length = 2 ** 16
    allowed_schemes: Optional[Collection[str]] = None
    tld_required: bool = False
    user_required: bool = False
    host_required: bool = True
    hidden_parts: Set[str] = set()
    __slots__ = ('scheme', 'user', 'password', 'host', 'tld', 'host_type', 'port', 'path', 'query', 'fragment')

    @no_type_check
    def __new__(cls, url: Optional[str], **kwargs) -> object:
        return str.__new__(cls, cls.build(**kwargs) if url is None else url)

    def __init__(self, url: str, *, scheme: str, user: Optional[str]=None, password: Optional[str]=None, host: Optional[str]=None, tld: Optional[str]=None, host_type: str='domain', port: Optional[str]=None, path: Optional[str]=None, query: Optional[str]=None, fragment: Optional[str]=None) -> None:
        str.__init__(url)
        self.scheme = scheme
        self.user = user
        self.password = password
        self.host = host
        self.tld = tld
        self.host_type = host_type
        self.port = port
        self.path = path
        self.query = query
        self.fragment = fragment

    @classmethod
    def build(cls, *, scheme: str, user: Optional[str]=None, password: Optional[str]=None, host: str, port: Optional[str]=None, path: Optional[str]=None, query: Optional[str]=None, fragment: Optional[str]=None, **_kwargs: str) -> str:
        parts = Parts(scheme=scheme, user=user, password=password, host=host, port=port, path=path, query=query, fragment=fragment, **_kwargs)
        url = scheme + '://'
        if user:
            url += user
        if password:
            url += ':' + password
        if user or password:
            url += '@'
        url += host
        if port and ('port' not in cls.hidden_parts or cls.get_default_parts(parts).get('port') != port):
            url += ':' + port
        if path:
            url += path
        if query:
            url += '?' + query
        if fragment:
            url += '#' + fragment
        return url

    @classmethod
    def __modify_schema__(cls, field_schema: Dict[str, Any]) -> None:
        update_not_none(field_schema, minLength=cls.min_length, maxLength=cls.max_length, format='uri')

    @classmethod
    def __get_validators__(cls) -> 'CallableGenerator':
        yield cls.validate

    @classmethod
    def validate(cls, value: Any, field: 'ModelField', config: 'BaseConfig') -> 'AnyUrl':
        if value.__class__ == cls:
            return value
        value = str_validator(value)
        if cls.strip_whitespace:
            value = value.strip()
        url: str = cast(str, constr_length_validator(value, field, config))
        m = cls._match_url(url)
        assert m, 'URL regex failed unexpectedly'
        original_parts = cast('Parts', m.groupdict())
        parts = cls.apply_default_parts(original_parts)
        parts = cls.validate_parts(parts)
        if m.end() != len(url):
            raise errors.UrlExtraError(extra=url[m.end():])
        return cls._build_url(m, url, parts)

    @classmethod
    def _build_url(cls, m: Match[str], url: str, parts: 'Parts') -> 'AnyUrl':
        """
        Validate hosts and build the AnyUrl object. Split from `validate` so this method
        can be altered in `MultiHostDsn`.
        """
        host, tld, host_type, rebuild = cls.validate_host(parts)
        return cls(None if rebuild else url, scheme=parts['scheme'], user=parts['user'], password=parts['password'], host=host, tld=tld, host_type=host_type, port=parts['port'], path=parts['path'], query=parts['query'], fragment=parts['fragment'])

    @staticmethod
    def _match_url(url: str) -> Optional[Match[str]]:
        return url_regex().match(url)

    @staticmethod
    def _validate_port(port: Optional[str]) -> None:
        if port is not None and int(port) > 65535:
            raise errors.UrlPortError()

    @classmethod
    def validate_parts(cls, parts: 'Parts', validate_port: bool=True) -> 'Parts':
        """
        A method used to validate parts of a URL.
        Could be overridden to set default values for parts if missing
        """
        scheme = parts['scheme']
        if scheme is None:
            raise errors.UrlSchemeError()
        if cls.allowed_schemes and scheme.lower() not in cls.allowed_schemes:
            raise errors.UrlSchemePermittedError(set(cls.allowed_schemes))
        if validate_port:
            cls._validate_port(parts['port'])
        user = parts['user']
        if cls.user_required and user is None:
            raise errors.UrlUserInfoError()
        return parts

    @classmethod
    def validate_host(cls, parts: 'Parts') -> Tuple[str, Optional[str], str, bool]:
        tld, host_type, rebuild = (None, None, False)
        for f in ('domain', 'ipv4', 'ipv6'):
            host = parts[f]
            if host:
                host_type = f
                break
        if host is None:
            if cls.host_required:
                raise errors.UrlHostError()
        elif host_type == 'domain':
            is_international = False
            d = ascii_domain_regex().fullmatch(host)
            if d is None:
                d = int_domain_regex().fullmatch(host)
                if d is None:
                    raise errors.UrlHostError()
                is_international = True
            tld = d.group('tld')
            if tld is None and (not is_international):
                d = int_domain_regex().fullmatch(host)
                assert d is not None
                tld = d.group('tld')
                is_international = True
            if tld is not None:
                tld = tld[1:]
            elif cls.tld_required:
                raise errors.UrlHostTldError()
            if is_international:
                host_type = 'int_domain'
                rebuild = True
                host = host.encode('idna').decode('ascii')
                if tld is not None:
                    tld = tld.encode('idna').decode('ascii')
        return (host, tld, host_type, rebuild)

    @staticmethod
    def get_default_parts(parts: 'Parts') -> 'Parts':
        return {}

    @classmethod
    def apply_default_parts(cls, parts: 'Parts') -> 'Parts':
        for key, value in cls.get_default_parts(parts).items():
            if not parts[key]:
                parts[key] = value
        return parts

    def __repr__(self) -> str:
        extra = ', '.join((f'{n}={getattr(self, n)!r}' for n in self.__slots__ if getattr(self, n) is not None))
        return f'{self.__class__.__name__}({super().__repr__()}, {extra})'