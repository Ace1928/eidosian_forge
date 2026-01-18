from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import abc
from googlecloudsdk.core.console import console_attr as ca
from googlecloudsdk.core.resource import resource_printer_base
import six
def _SetWidth(self, column_index, content_length):
    """Adjusts widths to account for the length of new column content.

    Args:
      column_index: The column index to potentially update. Must be between 0
        and len(widths).
      content_length: The column content's length to consider when updating
        widths.
    """
    if column_index == len(self._widths):
        self._widths.append(0)
    new_width = max(self._widths[column_index], content_length)
    if self._max_column_width is not None:
        new_width = min(self._max_column_width, new_width)
    self._widths[column_index] = new_width