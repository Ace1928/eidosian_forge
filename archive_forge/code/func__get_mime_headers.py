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
def _get_mime_headers(self) -> list[tuple[str, str]]:
    headers: list[tuple[str, str]] = []
    headers.append(('Project-Id-Version', f'{self.project} {self.version}'))
    headers.append(('Report-Msgid-Bugs-To', self.msgid_bugs_address))
    headers.append(('POT-Creation-Date', format_datetime(self.creation_date, 'yyyy-MM-dd HH:mmZ', locale='en')))
    if isinstance(self.revision_date, (datetime.datetime, datetime.time, int, float)):
        headers.append(('PO-Revision-Date', format_datetime(self.revision_date, 'yyyy-MM-dd HH:mmZ', locale='en')))
    else:
        headers.append(('PO-Revision-Date', self.revision_date))
    headers.append(('Last-Translator', self.last_translator))
    if self.locale_identifier:
        headers.append(('Language', str(self.locale_identifier)))
    if self.locale_identifier and 'LANGUAGE' in self.language_team:
        headers.append(('Language-Team', self.language_team.replace('LANGUAGE', str(self.locale_identifier))))
    else:
        headers.append(('Language-Team', self.language_team))
    if self.locale is not None:
        headers.append(('Plural-Forms', self.plural_forms))
    headers.append(('MIME-Version', '1.0'))
    headers.append(('Content-Type', f'text/plain; charset={self.charset}'))
    headers.append(('Content-Transfer-Encoding', '8bit'))
    headers.append(('Generated-By', f'Babel {VERSION}\n'))
    return headers