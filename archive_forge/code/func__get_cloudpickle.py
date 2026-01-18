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
def _get_cloudpickle() -> 'cloudpickle':
    return cast('cloudpickle', util.get_module('cloudpickle', 'ModelAdapter requires `cloudpickle`'))