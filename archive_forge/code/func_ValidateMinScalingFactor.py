from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import ipaddress
import re
from typing import Any
from googlecloudsdk.calliope import exceptions
def ValidateMinScalingFactor(scaling_factor):
    """Python hook to validate the min scaling factor."""
    return ValidateScalingFactorFloat(scaling_factor, '--min-scaling-factor')