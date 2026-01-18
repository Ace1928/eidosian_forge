import itertools
import numpy as np
from .._shared.utils import _supported_float_type, warn
from ..util import img_as_float
from . import rgb_colors
from .colorconv import gray2rgb, rgb2hsv, hsv2rgb
def _match_label_with_color(label, colors, bg_label, bg_color):
    """Return `unique_labels` and `color_cycle` for label array and color list.

    Colors are cycled for normal labels, but the background color should only
    be used for the background.
    """
    if bg_color is None:
        bg_color = (0, 0, 0)
    bg_color = _rgb_vector(bg_color)
    unique_labels, mapped_labels = np.unique(label, return_inverse=True)
    mapped_labels = mapped_labels.reshape(-1)
    bg_label_rank_list = mapped_labels[label.flat == bg_label]
    if len(bg_label_rank_list) > 0:
        bg_label_rank = bg_label_rank_list[0]
        mapped_labels[mapped_labels < bg_label_rank] += 1
        mapped_labels[label.flat == bg_label] = 0
    else:
        mapped_labels += 1
    color_cycle = itertools.cycle(colors)
    color_cycle = itertools.chain([bg_color], color_cycle)
    return (mapped_labels, color_cycle)