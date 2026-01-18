from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import io
from googlecloudsdk.core import log
from googlecloudsdk.core.resource import resource_printer_base
from googlecloudsdk.core.resource import resource_transform
import six
from six.moves import range  # pylint: disable=redefined-builtin
class _LiteralLines(six.text_type):
    """A yaml representer hook for literal strings containing newlines."""