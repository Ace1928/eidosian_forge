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
def _log_dataframe(self):
    x, y_true, y_pred = (None, None, None)
    if self.validation_data:
        x, y_true = (self.validation_data[0], self.validation_data[1])
        y_pred = self.model.predict(x)
    elif self.generator:
        if not self.validation_steps:
            wandb.termwarn('when using a generator for validation data with dataframes, you must pass validation_steps. skipping')
            return None
        for _ in range(self.validation_steps):
            bx, by_true = next(self.generator)
            by_pred = self.model.predict(bx)
            if x is None:
                x, y_true, y_pred = (bx, by_true, by_pred)
            else:
                x, y_true, y_pred = (np.append(x, bx, axis=0), np.append(y_true, by_true, axis=0), np.append(y_pred, by_pred, axis=0))
    if self.input_type in ('image', 'images') and self.output_type == 'label':
        return wandb.image_categorizer_dataframe(x=x, y_true=y_true, y_pred=y_pred, labels=self.labels)
    elif self.input_type in ('image', 'images') and self.output_type == 'segmentation_mask':
        return wandb.image_segmentation_dataframe(x=x, y_true=y_true, y_pred=y_pred, labels=self.labels, class_colors=self.class_colors)
    else:
        wandb.termwarn(f'unknown dataframe type for input_type={self.input_type} and output_type={self.output_type}')
        return None