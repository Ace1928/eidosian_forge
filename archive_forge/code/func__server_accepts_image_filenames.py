import hashlib
import logging
import os
from io import BytesIO
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Type, Union, cast
from urllib import parse
import wandb
from wandb import util
from wandb.sdk.lib import hashutil, runid
from wandb.sdk.lib.paths import LogicalPath
from ._private import MEDIA_TMP
from .base_types.media import BatchableMedia, Media
from .helper_types.bounding_boxes_2d import BoundingBoxes2D
from .helper_types.classes import Classes
from .helper_types.image_mask import ImageMask
def _server_accepts_image_filenames() -> bool:
    if util._is_offline():
        return True
    max_cli_version = util._get_max_cli_version()
    if max_cli_version is None:
        return False
    from wandb.util import parse_version
    accepts_image_filenames: bool = parse_version('0.12.10') <= parse_version(max_cli_version)
    return accepts_image_filenames