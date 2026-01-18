from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import dataclasses
from typing import Any, Dict, List, Union
from googlecloudsdk.core import exceptions
def _ParseThresholds(thresholds):
    if thresholds is None:
        return None
    return ScalingThresholds(scale_in=thresholds.scaleIn, scale_out=thresholds.scaleOut)