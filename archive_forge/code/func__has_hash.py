import numpy as np
import shutil
from ..util.dtype import img_as_bool
from ._registry import registry, registry_urls
from .. import __version__
import os.path as osp
import os
def _has_hash(path, expected_hash):
    """Check if the provided path has the expected hash."""
    if not osp.exists(path):
        return False
    return file_hash(path) == expected_hash