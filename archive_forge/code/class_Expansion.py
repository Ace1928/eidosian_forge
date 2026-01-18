from __future__ import annotations
import collections
from typing import Any, TYPE_CHECKING, cast
from .charset import Charset
from .variable import Variable
class Expansion:
    """
    Base class for template expansions.

    https://tools.ietf.org/html/rfc6570#section-3
    """

    def __init__(self) -> None:
        pass

    @property
    def variables(self) -> Iterable[Variable]:
        """Get all variables in this expansion."""
        return []

    @property
    def variable_names(self) -> Iterable[str]:
        """Get the names of all variables in this expansion."""
        return []

    def _encode(self, value: str, legal: str, pct_encoded: bool) -> str:
        """Encode a string into legal values."""
        output = ''
        index = 0
        while index < len(value):
            codepoint = value[index]
            if codepoint in legal:
                output += codepoint
            elif pct_encoded and '%' == codepoint and (index + 2 < len(value)) and (value[index + 1] in Charset.HEX_DIGIT) and (value[index + 2] in Charset.HEX_DIGIT):
                output += value[index:index + 3]
                index += 2
            else:
                utf8 = codepoint.encode('utf8')
                for byte in utf8:
                    output += '%' + Charset.HEX_DIGIT[int(byte / 16)] + Charset.HEX_DIGIT[byte % 16]
            index += 1
        return output

    def _uri_encode_value(self, value: str) -> str:
        """Encode a value into uri encoding."""
        return self._encode(value, Charset.UNRESERVED, False)

    def _uri_encode_name(self, name: str | int) -> str:
        """Encode a variable name into uri encoding."""
        return self._encode(str(name), Charset.UNRESERVED + Charset.RESERVED, True) if name else ''

    def _join(self, prefix: str, joiner: str, value: str) -> str:
        """Join a prefix to a value."""
        if prefix:
            return prefix + joiner + value
        return value

    def _encode_str(self, variable: Variable, name: str, value: str, prefix: str, joiner: str, first: bool) -> str:
        """Encode a string value for a variable."""
        if variable.max_length:
            if not first:
                raise ExpansionFailedError(str(variable))
            return self._join(prefix, joiner, self._uri_encode_value(value[:variable.max_length]))
        return self._join(prefix, joiner, self._uri_encode_value(value))

    def _encode_dict_item(self, variable: Variable, name: str, key: int | str, item: Any, delim: str, prefix: str, joiner: str, first: bool) -> str | None:
        """Encode a dict item for a variable."""
        joiner = '=' if variable.explode else ','
        if variable.array:
            name = self._uri_encode_name(key)
            prefix = prefix + '[' + name + ']' if prefix and (not first) else name
        else:
            prefix = self._join(prefix, '.', self._uri_encode_name(key))
        return self._encode_var(variable, str(key), item, delim, prefix, joiner, False)

    def _encode_list_item(self, variable: Variable, name: str, index: int, item: Any, delim: str, prefix: str, joiner: str, first: bool) -> str | None:
        """Encode a list item for a variable."""
        if variable.array:
            prefix = prefix + '[' + str(index) + ']' if prefix else ''
            return self._encode_var(variable, '', item, delim, prefix, joiner, False)
        return self._encode_var(variable, name, item, delim, prefix, '.', False)

    def _encode_var(self, variable: Variable, name: str, value: Any, delim: str=',', prefix: str='', joiner: str='=', first: bool=True) -> str | None:
        """Encode a variable."""
        if isinstance(value, str):
            return self._encode_str(variable, name, value, prefix, joiner, first)
        elif isinstance(value, collections.abc.Mapping):
            if len(value):
                encoded_items = [self._encode_dict_item(variable, name, key, value[key], delim, prefix, joiner, first) for key in value.keys()]
                return delim.join([item for item in encoded_items if item is not None])
            return None
        elif isinstance(value, collections.abc.Sequence):
            if len(value):
                encoded_items = [self._encode_list_item(variable, name, index, item, delim, prefix, joiner, first) for index, item in enumerate(value)]
                return delim.join([item for item in encoded_items if item is not None])
            return None
        elif isinstance(value, bool):
            return self._encode_str(variable, name, str(value).lower(), prefix, joiner, first)
        else:
            return self._encode_str(variable, name, str(value), prefix, joiner, first)

    def expand(self, values: Mapping[str, Any]) -> str | None:
        """Expand values."""
        return None

    def partial(self, values: Mapping[str, Any]) -> str:
        """Perform partial expansion."""
        return ''