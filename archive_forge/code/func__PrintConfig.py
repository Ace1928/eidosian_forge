from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import io
import shlex
from googlecloudsdk.core.resource import resource_printer_base
from googlecloudsdk.core.util import platforms
import six
def _PrintConfig(self, items):
    """Prints config items in the root category.

    Args:
      items: The current record dict iteritems from _AddRecord().
    """
    self._PrintCategory(self._out, [], items)