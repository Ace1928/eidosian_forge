from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import dataclasses
from googlecloudsdk.api_lib.accesscontextmanager import levels as levels_api
from googlecloudsdk.api_lib.accesscontextmanager import zones as perimeters_api
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.accesscontextmanager import policies
def GetLevelsQuotaUsage(self, levels_to_display):
    """Returns levels quota usage, only counts basic access levels.

    Args:
      levels_to_display: Response of ListAccessLevels API
    """
    access_levels = 0
    for level in levels_to_display:
        if level.basic:
            access_levels += 1
    return [Metric('Access levels', access_levels)]