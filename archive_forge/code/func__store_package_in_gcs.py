import asyncio
import hashlib
import logging
import os
import shutil
from enum import Enum
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Callable, List, Optional, Tuple
from urllib.parse import urlparse
from zipfile import ZipFile
from filelock import FileLock
from ray.util.annotations import DeveloperAPI
from ray._private.ray_constants import (
from ray._private.runtime_env.conda_utils import exec_cmd_stream_to_logger
from ray._private.thirdparty.pathspec import PathSpec
from ray.experimental.internal_kv import (
def _store_package_in_gcs(pkg_uri: str, data: bytes, logger: Optional[logging.Logger]=default_logger) -> int:
    """Stores package data in the Global Control Store (GCS).

    Args:
        pkg_uri: The GCS key to store the data in.
        data: The serialized package's bytes to store in the GCS.
        logger (Optional[logging.Logger]): The logger used by this function.

    Return:
        int: Size of data

    Raises:
        RuntimeError: If the upload to the GCS fails.
        ValueError: If the data's size exceeds GCS_STORAGE_MAX_SIZE.
    """
    file_size = len(data)
    size_str = _mib_string(file_size)
    if len(data) >= GCS_STORAGE_MAX_SIZE:
        raise ValueError(f"Package size ({size_str}) exceeds the maximum size of {_mib_string(GCS_STORAGE_MAX_SIZE)}. You can exclude large files using the 'excludes' option to the runtime_env or provide a remote URI of a zip file using protocols such as 's3://', 'https://' and so on, refer to https://docs.ray.io/en/latest/ray-core/handling-dependencies.html#api-reference.")
    logger.info(f"Pushing file package '{pkg_uri}' ({size_str}) to Ray cluster...")
    try:
        if os.environ.get(RAY_RUNTIME_ENV_FAIL_UPLOAD_FOR_TESTING_ENV_VAR):
            raise RuntimeError('Simulating failure to upload package for testing purposes.')
        _internal_kv_put(pkg_uri, data)
    except Exception as e:
        raise RuntimeError(f'Failed to store package in the GCS.\n  - GCS URI: {pkg_uri}\n  - Package data ({size_str}): {data[:15]}...\n') from e
    logger.info(f"Successfully pushed file package '{pkg_uri}'.")
    return len(data)