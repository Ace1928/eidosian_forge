import sys
import warnings
import numpy as np
import pytest
from skimage._shared import testing
from skimage._shared.utils import (
@channel_as_last_axis(multichannel_output=False)
def _decorated_channel_axis_size(x, *, channel_axis=None):
    if channel_axis is None:
        return None
    assert channel_axis == -1
    return x.shape[-1]