import logging
import os
from io import BytesIO
from typing import TYPE_CHECKING, Any, Dict, Optional, Sequence, Type, Union
from wandb import util
from wandb.sdk.lib import filesystem, runid
from . import _dtypes
from ._private import MEDIA_TMP
from .base_types.media import BatchableMedia
def _prepare_video(self, video: 'np.ndarray') -> 'np.ndarray':
    """This logic was mostly taken from tensorboardX."""
    np = util.get_module('numpy', required='wandb.Video requires numpy when passing raw data. To get it, run "pip install numpy".')
    if video.ndim < 4:
        raise ValueError('Video must be atleast 4 dimensions: time, channels, height, width')
    if video.ndim == 4:
        video = video.reshape(1, *video.shape)
    b, t, c, h, w = video.shape
    if video.dtype != np.uint8:
        logging.warning('Converting video data to uint8')
        video = video.astype(np.uint8)

    def is_power2(num: int) -> bool:
        return num != 0 and num & num - 1 == 0
    if not is_power2(video.shape[0]):
        len_addition = int(2 ** video.shape[0].bit_length() - video.shape[0])
        video = np.concatenate((video, np.zeros(shape=(len_addition, t, c, h, w))), axis=0)
    n_rows = 2 ** ((b.bit_length() - 1) // 2)
    n_cols = video.shape[0] // n_rows
    video = np.reshape(video, newshape=(n_rows, n_cols, t, c, h, w))
    video = np.transpose(video, axes=(2, 0, 4, 1, 5, 3))
    video = np.reshape(video, newshape=(t, n_rows * h, n_cols * w, c))
    return video