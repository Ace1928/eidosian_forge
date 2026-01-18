import os
import re
import subprocess
import sys
import tarfile
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from typing import List, Optional, Sequence, Tuple
import yaml
import ray  # noqa: F401
from ray.autoscaler._private.cli_logger import cli_logger
from ray.autoscaler._private.providers import _get_node_provider
from ray.autoscaler.tags import NODE_KIND_HEAD, NODE_KIND_WORKER, TAG_RAY_NODE_KIND
import psutil
class Archive:
    """Archive object to collect and compress files into a single file.

    Objects of this class can be passed around to different data collection
    functions. These functions can use the :meth:`subdir` method to add
    files to a sub directory of the archive.

    """

    def __init__(self, file: Optional[str]=None):
        self.file = file or tempfile.mkstemp(prefix='ray_logs_', suffix='.tar.gz')[1]
        self.tar = None
        self._lock = threading.Lock()

    @property
    def is_open(self):
        return bool(self.tar)

    def open(self):
        self.tar = tarfile.open(self.file, 'w:gz')

    def close(self):
        self.tar.close()
        self.tar = None

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    @contextmanager
    def subdir(self, subdir: str, root: Optional[str]='/'):
        """Open a context to add files to the archive.

        Example:

            .. code-block:: python

                with Archive("file.tar.gz") as archive:
                    with archive.subdir("logfiles", root="/tmp/logs") as sd:
                        # Will be added as `logfiles/nested/file.txt`
                        sd.add("/tmp/logs/nested/file.txt")

        Args:
            subdir: Subdir to which to add files to. Calling the
                ``add(path)`` command will place files into the ``subdir``
                directory of the archive.
            root: Root path. Files without an explicit ``arcname``
                will be named relatively to this path.

        Yields:
            A context object that can be used to add files to the archive.
        """
        root = os.path.abspath(root)

        class _Context:

            @staticmethod
            def add(path: str, arcname: Optional[str]=None):
                path = os.path.abspath(path)
                arcname = arcname or os.path.join(subdir, os.path.relpath(path, root))
                self._lock.acquire()
                self.tar.add(path, arcname=arcname)
                self._lock.release()
        yield _Context()