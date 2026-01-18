import math
from typing import Dict, List, Optional, Union
import numpy as np
from ...image_processing_utils import BaseImageProcessor, BatchFeature
from ...image_transforms import (
from ...image_utils import (
from ...utils import (
def _safe_convert_tensor(elem):
    try:
        return _convert_tensor(elem)
    except:
        if key == 'overflowing_values':
            raise ValueError('Unable to create tensor returning overflowing values of different lengths. ')
        raise ValueError("Unable to create tensor, you should probably activate padding with 'padding=True' to have batched tensors with the same length.")