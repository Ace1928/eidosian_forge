from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import abc
from googlecloudsdk.core.console import console_attr as ca
from googlecloudsdk.core.resource import resource_printer_base
import six
def _ProcessColumn(self, index, row):
    """Processes a single column value when computing column widths."""
    record = row[index]
    last_index = len(row) - 1
    if isinstance(record, _Marker):
        if index == last_index:
            self._MergeColumnWidths(record.CalculateColumnWidths(self._max_column_width, self._indent_length + INDENT_STEP))
            return
        else:
            raise TypeError('Markers can only be used in the last column.')
    if _IsLastColumnInRow(row, index, last_index, self._skip_empty):
        self._SetWidth(index, 0)
    else:
        console_attr = self._console_attr
        if self._console_attr is None:
            console_attr = ca.ConsoleAttr()
        width = console_attr.DisplayWidth(str(record)) + self._separator_width
        if index == 0:
            width += self._indent_length
        self._SetWidth(index, width)