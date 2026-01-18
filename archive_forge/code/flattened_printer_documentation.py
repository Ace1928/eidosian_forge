from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.core import properties
from googlecloudsdk.core.resource import resource_lex
from googlecloudsdk.core.resource import resource_printer_base
from googlecloudsdk.core.resource import resource_transform
import six
Immediately prints the record as flattened a flattened tree.

    Args:
      record: A JSON-serializable object.
      delimit: Prints resource delimiters if True.
    