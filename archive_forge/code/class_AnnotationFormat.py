import base64
import os
from io import BytesIO
from typing import TYPE_CHECKING, Dict, Iterable, List, Optional, Tuple, Union
import numpy as np
import requests
from packaging import version
from .utils import (
from .utils.constants import (  # noqa: F401
class AnnotationFormat(ExplicitEnum):
    COCO_DETECTION = 'coco_detection'
    COCO_PANOPTIC = 'coco_panoptic'