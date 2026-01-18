from __future__ import annotations
import os
import re
from collections.abc import Iterable
from typing import TYPE_CHECKING
from babel.core import Locale
from babel.messages.catalog import Catalog, Message
from babel.util import _cmp, wraptext
def _process_keyword_line(self, lineno, line, obsolete=False) -> None:
    for keyword in self._keywords:
        try:
            if line.startswith(keyword) and line[len(keyword)] in [' ', '[']:
                arg = line[len(keyword):]
                break
        except IndexError:
            self._invalid_pofile(line, lineno, 'Keyword must be followed by a string')
    else:
        self._invalid_pofile(line, lineno, "Start of line didn't match any expected keyword.")
        return
    if keyword in ['msgid', 'msgctxt']:
        self._finish_current_message()
    self.obsolete = obsolete
    if keyword == 'msgid':
        self.offset = lineno
    if keyword in ['msgid', 'msgid_plural']:
        self.in_msgctxt = False
        self.in_msgid = True
        self.messages.append(_NormalizedString(arg))
    elif keyword == 'msgstr':
        self.in_msgid = False
        self.in_msgstr = True
        if arg.startswith('['):
            idx, msg = arg[1:].split(']', 1)
            self.translations.append([int(idx), _NormalizedString(msg)])
        else:
            self.translations.append([0, _NormalizedString(arg)])
    elif keyword == 'msgctxt':
        self.in_msgctxt = True
        self.context = _NormalizedString(arg)