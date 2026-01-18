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
class Camera(TypedDict):
    viewpoint: Sequence[Point3D]
    target: Sequence[Point3D]