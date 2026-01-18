from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.core.resource import resource_printer_base
from googlecloudsdk.core.resource import resource_transform
from googlecloudsdk.core.util import encoding
import six
Prints the current record as CSV.

    Printer attributes:
      noheading: bool, Disable the initial key name heading record.

    Args:
      record: A list of JSON-serializable object columns.
      delimit: bool, Print resource delimiters -- ignored.

    Raises:
      ToolException: A data value has a type error.
    