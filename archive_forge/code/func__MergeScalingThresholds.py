from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import dataclasses
from typing import Any, Dict, List, Union
from googlecloudsdk.core import exceptions
def _MergeScalingThresholds(left: ScalingThresholds, right: ScalingThresholds) -> ScalingThresholds:
    """Merges two ScalingThresholds objects, favoring right one.

  Args:
    left: First ScalingThresholds object.
    right: Second ScalingThresholds object.

  Returns:
    Merged ScalingThresholds.
  """
    if left is None:
        return right
    if right is None:
        return left
    return ScalingThresholds(scale_in=_MergeFields(left.scale_in, right.scale_in), scale_out=_MergeFields(left.scale_out, right.scale_out))