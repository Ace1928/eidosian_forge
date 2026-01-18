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
def _save_model_as_artifact(self, epoch):
    if wandb.run.disabled:
        return
    self.model.save(self.filepath[:-3], overwrite=True, save_format='tf')
    name = wandb.util.make_artifact_name_safe(f'model-{wandb.run.name}')
    model_artifact = wandb.Artifact(name, type='model')
    model_artifact.add_dir(self.filepath[:-3])
    wandb.run.log_artifact(model_artifact, aliases=['latest', f'epoch_{epoch}'])
    shutil.rmtree(self.filepath[:-3])