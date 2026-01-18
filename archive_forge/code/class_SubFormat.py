from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import io
import json
import re
from googlecloudsdk.core import properties
from googlecloudsdk.core.console import console_attr
from googlecloudsdk.core.resource import resource_printer_base
from googlecloudsdk.core.resource import resource_projection_spec
from googlecloudsdk.core.resource import resource_transform
import six
from six.moves import range  # pylint: disable=redefined-builtin
class SubFormat(object):
    """A sub format object.

  Attributes:
    index: The parent column index.
    hidden: Column is projected but not displayed.
    printer: The nested printer object.
    out: The nested printer output stream.
    rows: The nested format aggregate rows if the parent has no columns.
    wrap: If column text should be wrapped.
  """

    def __init__(self, index, hidden, printer, out, wrap):
        self.index = index
        self.hidden = hidden
        self.printer = printer
        self.out = out
        self.rows = []
        self.wrap = wrap