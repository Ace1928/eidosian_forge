from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import display_taps
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import module_util
from googlecloudsdk.core import properties
from googlecloudsdk.core.cache import cache_update_ops
from googlecloudsdk.core.resource import resource_lex
from googlecloudsdk.core.resource import resource_printer
from googlecloudsdk.core.resource import resource_projection_spec
from googlecloudsdk.core.resource import resource_property
from googlecloudsdk.core.resource import resource_reference
from googlecloudsdk.core.resource import resource_transform
from googlecloudsdk.core.util import peek_iterable
import six
def _InitPrinter(self):
    """Initializes the printer and associated attributes."""
    if self._printer_is_initialized:
        return
    self._printer_is_initialized = True
    self._format = self.GetFormat()
    self._defaults = self._GetResourceInfoDefaults()
    if self._format:
        self._printer = resource_printer.Printer(self._format, defaults=self._defaults, out=log.out)
        if self._printer:
            self._defaults = self._printer.column_attributes