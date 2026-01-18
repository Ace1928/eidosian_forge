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
def _AddSortByTap(self):
    """Sorts the resources using the --sort-by keys."""
    if not resource_property.IsListLike(self._resources):
        return
    sort_keys = self._GetSortKeys()
    if not sort_keys:
        return
    self._args.sort_by = None
    groups = []
    group_keys = []
    group_reverse = False
    for key, reverse in sort_keys:
        if not group_keys:
            group_reverse = reverse
        elif group_reverse != reverse:
            groups.insert(0, (group_keys, group_reverse))
            group_keys = []
            group_reverse = reverse
        group_keys.append(key)
    if group_keys:
        groups.insert(0, (group_keys, group_reverse))
    for keys, reverse in groups:
        self._SortResources(keys, reverse)