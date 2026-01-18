from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import dataclasses
from typing import Any, Dict, List, Union
from googlecloudsdk.core import exceptions
def _AutoscalingThresholdsFromPolicy(policy: Dict[str, Union[str, int]], threshold: str) -> ScalingThresholds:
    scale_in = policy.get(f'{threshold}-scale-in')
    scale_out = policy.get(f'{threshold}-scale-out')
    if scale_in is None and scale_out is None:
        return None
    return ScalingThresholds(scale_in=scale_in, scale_out=scale_out)