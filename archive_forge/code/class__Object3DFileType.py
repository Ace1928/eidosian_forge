import codecs
import json
import os
import sys
from typing import (
import wandb
from wandb import util
from wandb.sdk.lib import runid
from wandb.sdk.lib.paths import LogicalPath
from . import _dtypes
from ._private import MEDIA_TMP
from .base_types.media import BatchableMedia
class _Object3DFileType(_dtypes.Type):
    name = 'object3D-file'
    types = [Object3D]