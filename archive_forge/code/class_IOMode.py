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
class IOMode(str, enum.Enum):
    """Available Image modes

    This is a helper enum for ``Request.Mode`` which is a composite of a
    ``Request.ImageMode`` and ``Request.IOMode``. The IOMode that tells the
    plugin if the resource should be read from or written to. Available values are

    - read ("r"): Read from the specified resource
    - write ("w"): Write to the specified resource

    """
    read = 'r'
    write = 'w'