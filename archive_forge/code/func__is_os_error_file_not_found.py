import os
import re
import urllib
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional
from ray._private.client_mode_hook import client_mode_hook
from ray._private.utils import _add_creatable_buckets_param_if_s3_uri, load_class
from ray._private.auto_init_hook import wrap_auto_init
def _is_os_error_file_not_found(err: OSError) -> bool:
    """Instead of "FileNotFoundError", pyarrow S3 filesystem raises
    OSError starts with "Path does not exist" for some of its APIs.

    # TODO(suquark): Delete this function after pyarrow handles missing files
    in a consistent way.
    """
    return len(err.args) > 0 and isinstance(err.args[0], str) and err.args[0].startswith('Path does not exist')