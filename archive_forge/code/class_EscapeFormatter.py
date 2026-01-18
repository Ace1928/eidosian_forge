import functools
import re
import string
import sys
import typing as t
class EscapeFormatter(string.Formatter):
    __slots__ = ('escape',)

    def __init__(self, escape: t.Callable[[t.Any], Markup]) -> None:
        self.escape = escape
        super().__init__()

    def format_field(self, value: t.Any, format_spec: str) -> str:
        if hasattr(value, '__html_format__'):
            rv = value.__html_format__(format_spec)
        elif hasattr(value, '__html__'):
            if format_spec:
                raise ValueError(f'Format specifier {format_spec} given, but {type(value)} does not define __html_format__. A class that defines __html__ must define __html_format__ to work with format specifiers.')
            rv = value.__html__()
        else:
            rv = string.Formatter.format_field(self, value, str(format_spec))
        return str(self.escape(rv))