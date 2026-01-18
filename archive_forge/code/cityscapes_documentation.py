import json
import os
from collections import namedtuple
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from PIL import Image
from .utils import extract_archive, iterable_to_str, verify_str_arg
from .vision import VisionDataset

        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is a tuple of all target types if target_type is a list with more
            than one item. Otherwise, target is a json object if target_type="polygon", else the image segmentation.
        