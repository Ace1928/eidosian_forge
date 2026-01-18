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
def _GetResourceInfoDefaults(self):
    """Returns the default symbols for --filter and --format."""
    if not self._info:
        return self._defaults
    symbols = self._info.GetTransforms()
    if not symbols and (not self._info.defaults):
        return self._defaults
    return resource_projection_spec.ProjectionSpec(defaults=resource_projection_spec.CombineDefaults([self._info.defaults, self._defaults]), symbols=symbols)