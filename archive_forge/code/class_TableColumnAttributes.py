from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
import io
from googlecloudsdk.core import log
from googlecloudsdk.core.resource import resource_printer
import six
from six.moves import range  # pylint: disable=redefined-builtin
class TableColumnAttributes(object):
    """Markdown table column attributes.

  Attributes:
    align: Column alignment, one of {'left', 'center', 'right'}.
    label: Column heading label string.
    width: Minimum column width.
  """

    def __init__(self, align='left', label=None, width=0):
        self.align = align
        self.label = label
        self.width = width