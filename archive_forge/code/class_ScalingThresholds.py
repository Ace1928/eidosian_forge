from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import dataclasses
from typing import Any, Dict, List, Union
from googlecloudsdk.core import exceptions
@dataclasses.dataclass(frozen=True)
class ScalingThresholds:
    """Scaling thresholds for a single condition. Uses None for empty values.

  Attributes:
    scale_in: The threshold for scaling in.
    scale_out: The threshold for scaling out.
  """
    scale_in: int
    scale_out: int