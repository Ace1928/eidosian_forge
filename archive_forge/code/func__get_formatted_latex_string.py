from __future__ import annotations
import io
import re
from html.parser import HTMLParser
from typing import Any
def _get_formatted_latex_string(self, options):
    lines = []
    wanted_fields = []
    if options['fields']:
        wanted_fields = [field for field in self._field_names if field in options['fields']]
    else:
        wanted_fields = self._field_names
    wanted_alignments = [self._align[field] for field in wanted_fields]
    if options['border'] and options['vrules'] == ALL:
        alignment_str = '|'.join(wanted_alignments)
    elif not options['border'] and options['preserve_internal_border']:
        alignment_str = '|'.join(wanted_alignments)
    else:
        alignment_str = ''.join(wanted_alignments)
    if options['border'] and options['vrules'] in [ALL, FRAME]:
        alignment_str = '|' + alignment_str + '|'
    begin_cmd = '\\begin{tabular}{%s}' % alignment_str
    lines.append(begin_cmd)
    if options['border'] and options['hrules'] in [ALL, FRAME]:
        lines.append('\\hline')
    if options['header']:
        lines.append(' & '.join(wanted_fields) + ' \\\\')
    if (options['border'] or options['preserve_internal_border']) and options['hrules'] in [ALL, HEADER]:
        lines.append('\\hline')
    rows = self._get_rows(options)
    formatted_rows = self._format_rows(rows)
    rows = self._get_rows(options)
    for row in formatted_rows:
        wanted_data = [d for f, d in zip(self._field_names, row) if f in wanted_fields]
        lines.append(' & '.join(wanted_data) + ' \\\\')
        if options['border'] and options['hrules'] == ALL:
            lines.append('\\hline')
    if options['border'] and options['hrules'] == FRAME:
        lines.append('\\hline')
    lines.append('\\end{tabular}')
    return '\r\n'.join(lines)