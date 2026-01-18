import logging
import operator
import os
import shutil
import sys
from itertools import chain
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K  # noqa: N812
import wandb
from wandb.sdk.integration_utils.data_logging import ValidationDataLogger
from wandb.sdk.lib.deprecate import Deprecated, deprecate
from wandb.util import add_import_hook
def _masks_to_pixels(self, masks):
    if len(masks[0].shape) == 2 or masks[0].shape[-1] == 1:
        return masks
    class_colors = self.class_colors if self.class_colors is not None else np.array(wandb.util.class_colors(masks[0].shape[2]))
    imgs = class_colors[np.argmax(masks, axis=-1)]
    return imgs