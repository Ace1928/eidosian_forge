import os
from io import BytesIO
import zipfile
import tempfile
import shutil
import enum
import warnings
from ..core import urlopen, get_remote_file
from pathlib import Path
from urllib.parse import urlparse
from typing import Optional
class ImageMode(str, enum.Enum):
    """Available Image modes

    This is a helper enum for ``Request.Mode`` which is a composite of a
    ``Request.ImageMode`` and ``Request.IOMode``. The image mode that tells the
    plugin the desired (and expected) image shape. Available values are

    - single_image ("i"): Return a single image extending in two spacial
      dimensions
    - multi_image ("I"): Return a list of images extending in two spacial
      dimensions
    - single_volume ("v"): Return an image extending into multiple dimensions.
      E.g. three spacial dimensions for image stacks, or two spatial and one
      time dimension for videos
    - multi_volume ("V"): Return a list of images extending into multiple
      dimensions.
    - any_mode ("?"): Return an image in any format (the plugin decides the
      appropriate action).

    """
    single_image = 'i'
    multi_image = 'I'
    single_volume = 'v'
    multi_volume = 'V'
    any_mode = '?'