import os
import re
import urllib
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional
from ray._private.client_mode_hook import client_mode_hook
from ray._private.utils import _add_creatable_buckets_param_if_s3_uri, load_class
from ray._private.auto_init_hook import wrap_auto_init
def _init_storage(storage_uri: str, is_head: bool):
    """Init global storage.

    On the head (ray start) process, this also creates a _valid file under the given
    storage path to validate the storage is writable. This file is also checked on each
    worker process to validate the storage is readable. This catches common errors
    like using a non-NFS filesystem path on a multi-node cluster.

    On worker nodes, the actual filesystem is lazily initialized on first use.
    """
    global _storage_uri
    if storage_uri:
        _storage_uri = storage_uri
        if is_head:
            _init_filesystem(create_valid_file=True)