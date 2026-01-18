from __future__ import annotations
import os
import re
from collections.abc import Iterable
from typing import TYPE_CHECKING
from babel.core import Locale
from babel.messages.catalog import Catalog, Message
from babel.util import _cmp, wraptext
def _process_comment(self, line) -> None:
    self._finish_current_message()
    if line[1:].startswith(':'):
        for location in line[2:].lstrip().split():
            pos = location.rfind(':')
            if pos >= 0:
                try:
                    lineno = int(location[pos + 1:])
                except ValueError:
                    continue
                self.locations.append((location[:pos], lineno))
            else:
                self.locations.append((location, None))
    elif line[1:].startswith(','):
        for flag in line[2:].lstrip().split(','):
            self.flags.append(flag.strip())
    elif line[1:].startswith('.'):
        comment = line[2:].strip()
        if comment:
            self.auto_comments.append(comment)
    else:
        self.user_comments.append(line[1:].strip())