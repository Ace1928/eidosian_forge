from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
import io
from googlecloudsdk.core import log
from googlecloudsdk.core.resource import resource_printer
import six
from six.moves import range  # pylint: disable=redefined-builtin
def SetCommand(self, command):
    """Sets the document command name.

    Args:
      command: The command split into component names.
    """
    self._command = command