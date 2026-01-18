from __future__ import absolute_import
from __future__ import annotations
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import collections
from collections.abc import Container, Mapping
from googlecloudsdk.core import exceptions
def _GetUnassignedTargets(self, new_percentages):
    """Get TrafficTargets with traffic not in new_percentages."""
    result = {}
    for target in self._m:
        key = GetKey(target)
        if target.percent and key not in new_percentages:
            result[key] = target
    return result