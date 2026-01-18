import errno
import os
from io import BytesIO
from typing import Set
import breezy
from .lazy_import import lazy_import
from breezy import (
from . import bedding
def _set_user_ignores(patterns):
    """Fill out the user ignore file with the given patterns

    This may raise an error if it doesn't have permission to
    write to the user ignore file.
    This is mostly used for testing, since it would be
    bad form to rewrite a user's ignore list.
    breezy only writes this file if it does not exist.
    """
    ignore_path = bedding.user_ignore_config_path()
    bedding.ensure_config_dir_exists()
    with open(ignore_path, 'wb') as f:
        for pattern in patterns:
            f.write(pattern.encode('utf8') + b'\n')