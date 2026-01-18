from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import ipaddress
import re
from typing import Any
from googlecloudsdk.calliope import exceptions
def ValidateScalingFactorFloat(scaling_factor, flag_name):
    """Validate the scaling factor float value."""
    if scaling_factor < 0.1 or scaling_factor > 6:
        raise exceptions.BadArgumentException(flag_name, 'Scaling factor ({0}) is not in the range [0.1, 6.0].'.format(scaling_factor))
    elif scaling_factor < 1 and scaling_factor * 10 % 1 != 0:
        raise exceptions.BadArgumentException(flag_name, 'Scaling factor less than 1.0 ({0}) should be a multiple of 0.1 (e.g. (0.1, 0.2, 0.3))'.format(scaling_factor))
    elif scaling_factor >= 1 and scaling_factor % 1.0 != 0:
        raise exceptions.BadArgumentException(flag_name, 'Scaling greater than 1.0 ({0}) should be a multiple of 1.0 (e.g. (1.0, 2.0, 3.0))'.format(scaling_factor))
    return scaling_factor