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
def _AddFilterTap(self):
    """Taps a resource filter into self.resources if needed."""
    expression = self._GetFilter()
    if not expression:
        return
    tap = display_taps.Filterer(expression, self._defaults)
    self._resources = peek_iterable.Tapper(self._resources, tap)