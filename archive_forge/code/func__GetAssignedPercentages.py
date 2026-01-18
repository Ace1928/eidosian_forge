from __future__ import absolute_import
from __future__ import annotations
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import collections
from collections.abc import Container, Mapping
from googlecloudsdk.core import exceptions
def _GetAssignedPercentages(self, new_percentages, unassigned_targets):
    percent_to_assign = self._GetPercentUnspecifiedTraffic(new_percentages)
    if percent_to_assign == 0:
        return {}
    percent_to_assign_from = sum((target.percent for target in unassigned_targets.values()))
    assigned_percentages = {}
    for k in unassigned_targets:
        assigned_percentages[k] = unassigned_targets[k].percent * float(percent_to_assign) / percent_to_assign_from
    return assigned_percentages