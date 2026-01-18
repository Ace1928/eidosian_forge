from filelock import FileLock
import hashlib
import os
from pathlib import Path
import ray
FileLock wrapper that uses temporary file locks.

    The temporary directory that these locks are saved to can be configured via
    the `RAY_TMPDIR` environment variable.

    Args:
        path: The file path that this temporary file lock is used for.
            This will be used to generate the lockfile filename.
            Ex: For concurrent writes to a file, this is the common filepath
            that multiple processes are writing to.
        **kwargs: Additional keyword arguments to pass to the underlying `FileLock`.
    