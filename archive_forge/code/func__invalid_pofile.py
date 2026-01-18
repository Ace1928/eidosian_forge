from __future__ import annotations
import os
import re
from collections.abc import Iterable
from typing import TYPE_CHECKING
from babel.core import Locale
from babel.messages.catalog import Catalog, Message
from babel.util import _cmp, wraptext
def _invalid_pofile(self, line, lineno, msg) -> None:
    assert isinstance(line, str)
    if self.abort_invalid:
        raise PoFileError(msg, self.catalog, line, lineno)
    print('WARNING:', msg)
    print(f'WARNING: Problem on line {lineno + 1}: {line!r}')