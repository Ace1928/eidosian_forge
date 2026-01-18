import os
import shutil
import sys
from typing import (
import wandb
from wandb import util
from wandb.sdk.lib import runid
from wandb.sdk.lib.hashutil import md5_file_hex
from wandb.sdk.lib.paths import LogicalPath
from ._private import MEDIA_TMP
from .base_types.wb_value import WBValue
class _TensorflowKerasSavedModel(_SavedModel['tensorflow.keras.Model']):
    _log_type = 'tfkeras-model-file'
    _path_extension = ''

    @staticmethod
    def _deserialize(dir_or_file_path: str) -> 'tensorflow.keras.Model':
        return _get_tf_keras().models.load_model(dir_or_file_path)

    @staticmethod
    def _validate_obj(obj: Any) -> bool:
        return isinstance(obj, _get_tf_keras().models.Model)

    @staticmethod
    def _serialize(model_obj: 'tensorflow.keras.Model', dir_or_file_path: str) -> None:
        _get_tf_keras().models.save_model(model_obj, dir_or_file_path, include_optimizer=True)