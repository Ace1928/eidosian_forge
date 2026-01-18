import base64
import dataclasses
import datetime
import errno
import json
import os
import subprocess
import tempfile
import time
import typing
from typing import Optional
from tensorboard import version
from tensorboard.util import tb_logging
def _get_info_file_path():
    """Get path to info file for the current process.

    As with `_get_info_dir`, the info directory will be created if it
    does not exist.
    """
    return os.path.join(_get_info_dir(), 'pid-%d.info' % os.getpid())