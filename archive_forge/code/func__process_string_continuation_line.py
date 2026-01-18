from __future__ import annotations
import os
import re
from collections.abc import Iterable
from typing import TYPE_CHECKING
from babel.core import Locale
from babel.messages.catalog import Catalog, Message
from babel.util import _cmp, wraptext
def _process_string_continuation_line(self, line, lineno) -> None:
    if self.in_msgid:
        s = self.messages[-1]
    elif self.in_msgstr:
        s = self.translations[-1][1]
    elif self.in_msgctxt:
        s = self.context
    else:
        self._invalid_pofile(line, lineno, 'Got line starting with " but not in msgid, msgstr or msgctxt')
        return
    s.append(line)