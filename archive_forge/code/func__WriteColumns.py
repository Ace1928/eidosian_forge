from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import abc
from googlecloudsdk.core.console import console_attr as ca
from googlecloudsdk.core.resource import resource_printer_base
import six
def _WriteColumns(self, output, indent_length, column_values):
    """Writes generated column values to output with the given indentation."""
    output.write(_GenerateLineValue(self._COLUMN_SEPARATOR.join(column_values).rstrip(), indent_length))