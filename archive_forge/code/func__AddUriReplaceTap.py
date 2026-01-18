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
def _AddUriReplaceTap(self):
    """Taps a resource Uri replacer into self.resources if needed."""
    if not self._GetFlag('uri'):
        return
    tap = display_taps.UriReplacer(self._transform_uri)
    self._resources = peek_iterable.Tapper(self._resources, tap)