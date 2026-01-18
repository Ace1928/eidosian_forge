from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import os
import sys
from googlecloudsdk.core.console import console_attr
from six.moves import range  # pylint: disable=redefined-builtin
def _Dimensions(items):
    """Returns the transpose dimensions for items."""
    longest_item_len = max((len(x) for x in items))
    column_count = int(width / (len(pad) + longest_item_len)) or 1
    row_count = _IntegerCeilingDivide(len(items), column_count)
    return (longest_item_len, column_count, row_count)