from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import abc
from googlecloudsdk.core.console import console_attr as ca
from googlecloudsdk.core.resource import resource_printer_base
import six
def _GenerateColumnValue(self, index, row, indent_length, column_widths):
    """Generates the string value for one column in one row."""
    width = column_widths.widths[index]
    if index == 0:
        width -= indent_length
    separator = self.separator if index < len(row) - 1 else ''
    console_attr = self._console_attr
    if self._console_attr is None:
        console_attr = ca.ConsoleAttr()
    n_padding = width - console_attr.DisplayWidth(str(row[index])) - len(separator)
    return str(row[index]) + separator + n_padding * ' '