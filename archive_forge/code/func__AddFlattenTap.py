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
def _AddFlattenTap(self):
    """Taps one or more resource flatteners into self.resources if needed."""

    def _Slice(key):
        """Helper to add one flattened slice tap."""
        tap = display_taps.Flattener(key)
        self._resources = peek_iterable.Tapper(self._resources, tap)
    keys = self._GetFlatten()
    if not keys:
        return
    for key in keys:
        flattened_key = []
        sliced = False
        for k in resource_lex.Lexer(key).Key():
            if k is None:
                sliced = True
                _Slice(flattened_key)
            else:
                sliced = False
                flattened_key.append(k)
        if not sliced:
            _Slice(flattened_key)