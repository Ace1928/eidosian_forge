from __future__ import annotations
import base64
import binascii
import typing as t
from ..http import dump_header
from ..http import parse_dict_header
from ..http import quote_header_value
from .structures import CallbackDict
class Authorization:
    """Represents the parts of an ``Authorization`` request header.

    :attr:`.Request.authorization` returns an instance if the header is set.

    An instance can be used with the test :class:`.Client` request methods' ``auth``
    parameter to send the header in test requests.

    Depending on the auth scheme, either :attr:`parameters` or :attr:`token` will be
    set. The ``Basic`` scheme's token is decoded into the ``username`` and ``password``
    parameters.

    For convenience, ``auth["key"]`` and ``auth.key`` both access the key in the
    :attr:`parameters` dict, along with ``auth.get("key")`` and ``"key" in auth``.

    .. versionchanged:: 2.3
        The ``token`` parameter and attribute was added to support auth schemes that use
        a token instead of parameters, such as ``Bearer``.

    .. versionchanged:: 2.3
        The object is no longer a ``dict``.

    .. versionchanged:: 0.5
        The object is an immutable dict.
    """

    def __init__(self, auth_type: str, data: dict[str, str | None] | None=None, token: str | None=None) -> None:
        self.type = auth_type
        'The authorization scheme, like ``basic``, ``digest``, or ``bearer``.'
        if data is None:
            data = {}
        self.parameters = data
        'A dict of parameters parsed from the header. Either this or :attr:`token`\n        will have a value for a given scheme.\n        '
        self.token = token
        'A token parsed from the header. Either this or :attr:`parameters` will have a\n        value for a given scheme.\n\n        .. versionadded:: 2.3\n        '

    def __getattr__(self, name: str) -> str | None:
        return self.parameters.get(name)

    def __getitem__(self, name: str) -> str | None:
        return self.parameters.get(name)

    def get(self, key: str, default: str | None=None) -> str | None:
        return self.parameters.get(key, default)

    def __contains__(self, key: str) -> bool:
        return key in self.parameters

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Authorization):
            return NotImplemented
        return other.type == self.type and other.token == self.token and (other.parameters == self.parameters)

    @classmethod
    def from_header(cls, value: str | None) -> te.Self | None:
        """Parse an ``Authorization`` header value and return an instance, or ``None``
        if the value is empty.

        :param value: The header value to parse.

        .. versionadded:: 2.3
        """
        if not value:
            return None
        scheme, _, rest = value.partition(' ')
        scheme = scheme.lower()
        rest = rest.strip()
        if scheme == 'basic':
            try:
                username, _, password = base64.b64decode(rest).decode().partition(':')
            except (binascii.Error, UnicodeError):
                return None
            return cls(scheme, {'username': username, 'password': password})
        if '=' in rest.rstrip('='):
            return cls(scheme, parse_dict_header(rest), None)
        return cls(scheme, None, rest)

    def to_header(self) -> str:
        """Produce an ``Authorization`` header value representing this data.

        .. versionadded:: 2.0
        """
        if self.type == 'basic':
            value = base64.b64encode(f'{self.username}:{self.password}'.encode()).decode('utf8')
            return f'Basic {value}'
        if self.token is not None:
            return f'{self.type.title()} {self.token}'
        return f'{self.type.title()} {dump_header(self.parameters)}'

    def __str__(self) -> str:
        return self.to_header()

    def __repr__(self) -> str:
        return f'<{type(self).__name__} {self.to_header()}>'