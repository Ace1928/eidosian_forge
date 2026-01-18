from __future__ import annotations
import io
import re
from html.parser import HTMLParser
from typing import Any
def _compute_widths(self, rows, options) -> None:
    if options['header']:
        widths = [_get_size(field)[0] for field in self._field_names]
    else:
        widths = len(self.field_names) * [0]
    for row in rows:
        for index, value in enumerate(row):
            fieldname = self.field_names[index]
            if self.none_format.get(fieldname) is not None:
                if value == 'None' or value is None:
                    value = self.none_format.get(fieldname)
            if fieldname in self.max_width:
                widths[index] = max(widths[index], min(_get_size(value)[0], self.max_width[fieldname]))
            else:
                widths[index] = max(widths[index], _get_size(value)[0])
            if fieldname in self.min_width:
                widths[index] = max(widths[index], self.min_width[fieldname])
    self._widths = widths
    if self._max_table_width:
        table_width = self._compute_table_width(options)
        if table_width > self._max_table_width:
            scale = 1.0 * self._max_table_width / table_width
            widths = [int(w * scale) for w in widths]
            self._widths = widths
    if self._min_table_width or options['title']:
        if options['title']:
            title_width = len(options['title']) + sum(self._get_padding_widths(options))
            if options['vrules'] in (FRAME, ALL):
                title_width += 2
        else:
            title_width = 0
        min_table_width = self.min_table_width or 0
        min_width = max(title_width, min_table_width)
        if options['border']:
            borders = len(widths) + 1
        elif options['preserve_internal_border']:
            borders = len(widths)
        else:
            borders = 0
        min_width -= sum([sum(self._get_padding_widths(options)) for _ in widths]) + borders
        content_width = sum(widths) or 1
        if content_width < min_width:
            scale = 1.0 * min_width / content_width
            widths = [int(w * scale) for w in widths]
            if sum(widths) < min_width:
                widths[-1] += min_width - sum(widths)
            self._widths = widths