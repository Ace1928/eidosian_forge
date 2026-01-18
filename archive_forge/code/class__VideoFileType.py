import logging
import os
from io import BytesIO
from typing import TYPE_CHECKING, Any, Dict, Optional, Sequence, Type, Union
from wandb import util
from wandb.sdk.lib import filesystem, runid
from . import _dtypes
from ._private import MEDIA_TMP
from .base_types.media import BatchableMedia
class _VideoFileType(_dtypes.Type):
    name = 'video-file'
    types = [Video]