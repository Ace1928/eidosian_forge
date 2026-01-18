import re
import sys
from datetime import datetime, timedelta
from datetime import tzinfo as dt_tzinfo
from functools import lru_cache
from typing import (
from dateutil import tz
from arrow import locales
from arrow.constants import DEFAULT_LOCALE
from arrow.util import next_weekday, normalize_timestamp
def _generate_pattern_re(self, fmt: str) -> Tuple[List[_FORMAT_TYPE], Pattern[str]]:
    tokens: List[_FORMAT_TYPE] = []
    offset = 0
    escaped_fmt = re.escape(fmt)
    escaped_fmt = re.sub(self._ESCAPE_RE, '#', escaped_fmt)
    escaped_fmt = re.sub('S+', 'S', escaped_fmt)
    escaped_data = re.findall(self._ESCAPE_RE, fmt)
    fmt_pattern = escaped_fmt
    for m in self._FORMAT_RE.finditer(escaped_fmt):
        token: _FORMAT_TYPE = cast(_FORMAT_TYPE, m.group(0))
        try:
            input_re = self._input_re_map[token]
        except KeyError:
            raise ParserError(f'Unrecognized token {token!r}.')
        input_pattern = f'(?P<{token}>{input_re.pattern})'
        tokens.append(token)
        fmt_pattern = fmt_pattern[:m.start() + offset] + input_pattern + fmt_pattern[m.end() + offset:]
        offset += len(input_pattern) - (m.end() - m.start())
    final_fmt_pattern = ''
    split_fmt = fmt_pattern.split('\\#')
    for i in range(len(split_fmt)):
        final_fmt_pattern += split_fmt[i]
        if i < len(escaped_data):
            final_fmt_pattern += escaped_data[i][1:-1]
    starting_word_boundary = '(?<!\\S\\S)(?<![^\\,\\.\\;\\:\\?\\!\\"\\\'\\`\\[\\]\\{\\}\\(\\)<>\\s])(\\b|^)'
    ending_word_boundary = '(?=[\\,\\.\\;\\:\\?\\!\\"\\\'\\`\\[\\]\\{\\}\\(\\)\\<\\>]?(?!\\S))'
    bounded_fmt_pattern = '{}{}{}'.format(starting_word_boundary, final_fmt_pattern, ending_word_boundary)
    return (tokens, re.compile(bounded_fmt_pattern, flags=re.IGNORECASE))