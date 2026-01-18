from __future__ import absolute_import
from __future__ import annotations
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import collections
from collections.abc import Container, Mapping
from googlecloudsdk.core import exceptions
def _GetPercentUnspecifiedTraffic(self, new_percentages):
    """Returns percentage of traffic not explicitly specified by caller."""
    specified_percent = sum(new_percentages.values())
    return 100 - specified_percent