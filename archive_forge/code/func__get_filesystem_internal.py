import os
import re
import urllib
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional
from ray._private.client_mode_hook import client_mode_hook
from ray._private.utils import _add_creatable_buckets_param_if_s3_uri, load_class
from ray._private.auto_init_hook import wrap_auto_init
def _get_filesystem_internal() -> ('pyarrow.fs.FileSystem', str):
    """Internal version of get_filesystem() that doesn't hit Ray client hooks.

    This forces full (non-lazy) init of the filesystem.
    """
    global _filesystem, _storage_prefix
    if _filesystem is None:
        _init_filesystem()
    return (_filesystem, _storage_prefix)