import atexit
import collections
import datetime
import errno
import json
import logging
import os
import random
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
import traceback
from collections import defaultdict
from typing import Dict, Optional, Tuple, IO, AnyStr
from filelock import FileLock
import ray
import ray._private.ray_constants as ray_constants
import ray._private.services
from ray._private import storage
from ray._raylet import GcsClient, get_session_key_from_storage
from ray._private.resource_spec import ResourceSpec
from ray._private.services import serialize_config, get_address
from ray._private.utils import open_log, try_to_create_directory, try_to_symlink
def _make_inc_temp(self, suffix: str='', prefix: str='', directory_name: Optional[str]=None):
    """Return an incremental temporary file name. The file is not created.

        Args:
            suffix: The suffix of the temp file.
            prefix: The prefix of the temp file.
            directory_name (str) : The base directory of the temp file.

        Returns:
            A string of file name. If there existing a file having
                the same name, the returned name will look like
                "{directory_name}/{prefix}.{unique_index}{suffix}"
        """
    if directory_name is None:
        directory_name = ray._private.utils.get_ray_temp_dir()
    directory_name = os.path.expanduser(directory_name)
    index = self._incremental_dict[suffix, prefix, directory_name]
    while index < tempfile.TMP_MAX:
        if index == 0:
            filename = os.path.join(directory_name, prefix + suffix)
        else:
            filename = os.path.join(directory_name, prefix + '.' + str(index) + suffix)
        index += 1
        if not os.path.exists(filename):
            self._incremental_dict[suffix, prefix, directory_name] = index
            return filename
    raise FileExistsError(errno.EEXIST, 'No usable temporary filename found')