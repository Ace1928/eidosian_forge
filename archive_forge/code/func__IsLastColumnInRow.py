from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import abc
from googlecloudsdk.core.console import console_attr as ca
from googlecloudsdk.core.resource import resource_printer_base
import six
def _IsLastColumnInRow(row, column_index, last_index, skip_empty):
    """Returns true if column_index is considered the last column in the row."""
    return column_index == last_index or (skip_empty and _FollowedByEmpty(row, column_index)) or isinstance(row[column_index + 1], _Marker)