from __future__ import annotations
import datetime
import re
import string
from tomlkit._compat import decode
from tomlkit._utils import RFC_3339_LOOSE
from tomlkit._utils import _escaped
from tomlkit._utils import parse_rfc3339
from tomlkit.container import Container
from tomlkit.exceptions import EmptyKeyError
from tomlkit.exceptions import EmptyTableNameError
from tomlkit.exceptions import InternalParserError
from tomlkit.exceptions import InvalidCharInStringError
from tomlkit.exceptions import InvalidControlChar
from tomlkit.exceptions import InvalidDateError
from tomlkit.exceptions import InvalidDateTimeError
from tomlkit.exceptions import InvalidNumberError
from tomlkit.exceptions import InvalidTimeError
from tomlkit.exceptions import InvalidUnicodeValueError
from tomlkit.exceptions import ParseError
from tomlkit.exceptions import UnexpectedCharError
from tomlkit.exceptions import UnexpectedEofError
from tomlkit.items import AoT
from tomlkit.items import Array
from tomlkit.items import Bool
from tomlkit.items import BoolType
from tomlkit.items import Comment
from tomlkit.items import Date
from tomlkit.items import DateTime
from tomlkit.items import Float
from tomlkit.items import InlineTable
from tomlkit.items import Integer
from tomlkit.items import Item
from tomlkit.items import Key
from tomlkit.items import KeyType
from tomlkit.items import Null
from tomlkit.items import SingleKey
from tomlkit.items import String
from tomlkit.items import StringType
from tomlkit.items import Table
from tomlkit.items import Time
from tomlkit.items import Trivia
from tomlkit.items import Whitespace
from tomlkit.source import Source
from tomlkit.toml_char import TOMLChar
from tomlkit.toml_document import TOMLDocument
def _parse_escaped_char(self, multiline):
    if multiline and self._current.is_ws():
        tmp = ''
        while self._current.is_ws():
            tmp += self._current
            self.inc(exception=UnexpectedEofError)
            continue
        if '\n' not in tmp:
            raise self.parse_error(InvalidCharInStringError, self._current)
        return ''
    if self._current in _escaped:
        c = _escaped[self._current]
        self.inc(exception=UnexpectedEofError)
        return c
    if self._current in {'u', 'U'}:
        u, ue = self._peek_unicode(self._current == 'U')
        if u is not None:
            self.inc_n(len(ue) + 1)
            return u
        raise self.parse_error(InvalidUnicodeValueError)
    raise self.parse_error(InvalidCharInStringError, self._current)