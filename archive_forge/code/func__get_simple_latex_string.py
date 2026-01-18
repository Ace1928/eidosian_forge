from __future__ import annotations
import io
import re
from html.parser import HTMLParser
from typing import Any
def _get_simple_latex_string(self, options):
    lines = []
    wanted_fields = []
    if options['fields']:
        wanted_fields = [field for field in self._field_names if field in options['fields']]
    else:
        wanted_fields = self._field_names
    alignments = ''.join([self._align[field] for field in wanted_fields])
    begin_cmd = '\\begin{tabular}{%s}' % alignments
    lines.append(begin_cmd)
    if options['header']:
        lines.append(' & '.join(wanted_fields) + ' \\\\')
    rows = self._get_rows(options)
    formatted_rows = self._format_rows(rows)
    for row in formatted_rows:
        wanted_data = [d for f, d in zip(self._field_names, row) if f in wanted_fields]
        lines.append(' & '.join(wanted_data) + ' \\\\')
    lines.append('\\end{tabular}')
    return '\r\n'.join(lines)