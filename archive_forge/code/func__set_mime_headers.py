from __future__ import annotations
import datetime
import re
from collections import OrderedDict
from collections.abc import Iterable, Iterator
from copy import copy
from difflib import SequenceMatcher
from email import message_from_string
from heapq import nlargest
from typing import TYPE_CHECKING
from babel import __version__ as VERSION
from babel.core import Locale, UnknownLocaleError
from babel.dates import format_datetime
from babel.messages.plurals import get_plural
from babel.util import LOCALTZ, FixedOffsetTimezone, _cmp, distinct
def _set_mime_headers(self, headers: Iterable[tuple[str, str]]) -> None:
    for name, value in headers:
        name = self._force_text(name.lower(), encoding=self.charset)
        value = self._force_text(value, encoding=self.charset)
        if name == 'project-id-version':
            parts = value.split(' ')
            self.project = ' '.join(parts[:-1])
            self.version = parts[-1]
        elif name == 'report-msgid-bugs-to':
            self.msgid_bugs_address = value
        elif name == 'last-translator':
            self.last_translator = value
        elif name == 'language':
            value = value.replace('-', '_')
            self._set_locale(value)
        elif name == 'language-team':
            self.language_team = value
        elif name == 'content-type':
            params = parse_separated_header(value)
            if 'charset' in params:
                self.charset = params['charset'].lower()
        elif name == 'plural-forms':
            params = parse_separated_header(f' ;{value}')
            self._num_plurals = int(params.get('nplurals', 2))
            self._plural_expr = params.get('plural', '(n != 1)')
        elif name == 'pot-creation-date':
            self.creation_date = _parse_datetime_header(value)
        elif name == 'po-revision-date':
            if 'YEAR' not in value:
                self.revision_date = _parse_datetime_header(value)