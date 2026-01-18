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
def _initialize_from_wbimage(self, wbimage: 'Image') -> None:
    self._grouping = wbimage._grouping
    self._caption = wbimage._caption
    self._width = wbimage._width
    self._height = wbimage._height
    self._image = wbimage._image
    self._classes = wbimage._classes
    self._path = wbimage._path
    self._is_tmp = wbimage._is_tmp
    self._extension = wbimage._extension
    self._sha256 = wbimage._sha256
    self._size = wbimage._size
    self.format = wbimage.format
    self._file_type = wbimage._file_type
    self._artifact_source = wbimage._artifact_source
    self._artifact_target = wbimage._artifact_target