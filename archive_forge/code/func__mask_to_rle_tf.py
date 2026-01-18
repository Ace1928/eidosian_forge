import math
from copy import deepcopy
from itertools import product
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from ...image_processing_utils import BaseImageProcessor, BatchFeature, get_size_dict
from ...image_transforms import convert_to_rgb, pad, resize, to_channel_dimension_format
from ...image_utils import (
from ...utils import (
def _mask_to_rle_tf(input_mask: 'tf.Tensor'):
    """
    Encodes masks the run-length encoding (RLE), in the format expected by pycoco tools.
    """
    batch_size, height, width = input_mask.shape
    input_mask = flatten(tf.transpose(input_mask, perm=(0, 2, 1)), 1)
    diff = input_mask[:, 1:] ^ input_mask[:, :-1]
    change_indices = tf.where(diff)
    out = []
    for i in range(batch_size):
        cur_idxs = change_indices[change_indices[:, 0] == i, 1] + 1
        btw_idxs = cur_idxs[1:] - cur_idxs[:-1]
        counts = [] if input_mask[i, 0] == 0 else [0]
        counts += [cur_idxs[0].item()] + btw_idxs.tolist() + [height * width - cur_idxs[-1]]
        out.append({'size': [height, width], 'counts': counts})
    return out