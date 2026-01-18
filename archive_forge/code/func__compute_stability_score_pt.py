import math
from copy import deepcopy
from itertools import product
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from ...image_processing_utils import BaseImageProcessor, BatchFeature, get_size_dict
from ...image_transforms import convert_to_rgb, pad, resize, to_channel_dimension_format
from ...image_utils import (
from ...utils import (
def _compute_stability_score_pt(masks: 'torch.Tensor', mask_threshold: float, stability_score_offset: int):
    intersections = (masks > mask_threshold + stability_score_offset).sum(-1, dtype=torch.int16).sum(-1, dtype=torch.int32)
    unions = (masks > mask_threshold - stability_score_offset).sum(-1, dtype=torch.int16).sum(-1, dtype=torch.int32)
    stability_scores = intersections / unions
    return stability_scores