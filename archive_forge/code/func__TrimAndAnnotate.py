from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import os
import sys
from googlecloudsdk.core.console import console_attr
from six.moves import range  # pylint: disable=redefined-builtin
def _TrimAndAnnotate(item, longest_item_len):
    """Truncates and appends '*' if len(item) > longest_item_len."""
    if len(item) <= longest_item_len:
        return item
    return item[:longest_item_len] + '*'