import contextlib
import glob
import json
import logging
import os
import platform
import shutil
import tempfile
import traceback
import uuid
from typing import Any, Dict, Iterator, List, Optional, Union
import pyarrow.fs
from ray.air._internal.filelock import TempFileLock
from ray.train._internal.storage import _download_from_fs_path, _exists_at_fs_path
from ray.util.annotations import PublicAPI
def _get_uri_error(name: str):
    return AttributeError(f'The new `ray.train.Checkpoint` class does not support `{name}()`. To create a checkpoint from remote storage, create a `Checkpoint` using its constructor instead of `from_directory`.\nExample: `Checkpoint(path="s3://a/b/c")`.\nThen, access the contents of the checkpoint with `checkpoint.as_directory()` / `checkpoint.to_directory()`.\nTo upload data to remote storage, use e.g. `pyarrow.fs.FileSystem` or your client of choice.')