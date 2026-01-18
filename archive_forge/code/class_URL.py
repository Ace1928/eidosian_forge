from __future__ import annotations
import typing
from shlex import shlex
from urllib.parse import SplitResult, parse_qsl, urlencode, urlsplit
from starlette.concurrency import run_in_threadpool
from starlette.types import Scope
class URL:

    def __init__(self, url: str='', scope: Scope | None=None, **components: typing.Any) -> None:
        if scope is not None:
            assert not url, 'Cannot set both "url" and "scope".'
            assert not components, 'Cannot set both "scope" and "**components".'
            scheme = scope.get('scheme', 'http')
            server = scope.get('server', None)
            path = scope['path']
            query_string = scope.get('query_string', b'')
            host_header = None
            for key, value in scope['headers']:
                if key == b'host':
                    host_header = value.decode('latin-1')
                    break
            if host_header is not None:
                url = f'{scheme}://{host_header}{path}'
            elif server is None:
                url = path
            else:
                host, port = server
                default_port = {'http': 80, 'https': 443, 'ws': 80, 'wss': 443}[scheme]
                if port == default_port:
                    url = f'{scheme}://{host}{path}'
                else:
                    url = f'{scheme}://{host}:{port}{path}'
            if query_string:
                url += '?' + query_string.decode()
        elif components:
            assert not url, 'Cannot set both "url" and "**components".'
            url = URL('').replace(**components).components.geturl()
        self._url = url

    @property
    def components(self) -> SplitResult:
        if not hasattr(self, '_components'):
            self._components = urlsplit(self._url)
        return self._components

    @property
    def scheme(self) -> str:
        return self.components.scheme

    @property
    def netloc(self) -> str:
        return self.components.netloc

    @property
    def path(self) -> str:
        return self.components.path

    @property
    def query(self) -> str:
        return self.components.query

    @property
    def fragment(self) -> str:
        return self.components.fragment

    @property
    def username(self) -> None | str:
        return self.components.username

    @property
    def password(self) -> None | str:
        return self.components.password

    @property
    def hostname(self) -> None | str:
        return self.components.hostname

    @property
    def port(self) -> int | None:
        return self.components.port

    @property
    def is_secure(self) -> bool:
        return self.scheme in ('https', 'wss')

    def replace(self, **kwargs: typing.Any) -> URL:
        if 'username' in kwargs or 'password' in kwargs or 'hostname' in kwargs or ('port' in kwargs):
            hostname = kwargs.pop('hostname', None)
            port = kwargs.pop('port', self.port)
            username = kwargs.pop('username', self.username)
            password = kwargs.pop('password', self.password)
            if hostname is None:
                netloc = self.netloc
                _, _, hostname = netloc.rpartition('@')
                if hostname[-1] != ']':
                    hostname = hostname.rsplit(':', 1)[0]
            netloc = hostname
            if port is not None:
                netloc += f':{port}'
            if username is not None:
                userpass = username
                if password is not None:
                    userpass += f':{password}'
                netloc = f'{userpass}@{netloc}'
            kwargs['netloc'] = netloc
        components = self.components._replace(**kwargs)
        return self.__class__(components.geturl())

    def include_query_params(self, **kwargs: typing.Any) -> URL:
        params = MultiDict(parse_qsl(self.query, keep_blank_values=True))
        params.update({str(key): str(value) for key, value in kwargs.items()})
        query = urlencode(params.multi_items())
        return self.replace(query=query)

    def replace_query_params(self, **kwargs: typing.Any) -> URL:
        query = urlencode([(str(key), str(value)) for key, value in kwargs.items()])
        return self.replace(query=query)

    def remove_query_params(self, keys: str | typing.Sequence[str]) -> 'URL':
        if isinstance(keys, str):
            keys = [keys]
        params = MultiDict(parse_qsl(self.query, keep_blank_values=True))
        for key in keys:
            params.pop(key, None)
        query = urlencode(params.multi_items())
        return self.replace(query=query)

    def __eq__(self, other: typing.Any) -> bool:
        return str(self) == str(other)

    def __str__(self) -> str:
        return self._url

    def __repr__(self) -> str:
        url = str(self)
        if self.password:
            url = str(self.replace(password='********'))
        return f'{self.__class__.__name__}({repr(url)})'