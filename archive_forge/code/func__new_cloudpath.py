import abc
from collections import defaultdict
import collections.abc
from contextlib import contextmanager
import os
from pathlib import (  # type: ignore
import shutil
import sys
from typing import (
from urllib.parse import urlparse
from warnings import warn
from cloudpathlib.enums import FileCacheMode
from . import anypath
from .exceptions import (
def _new_cloudpath(self, path: Union[str, os.PathLike]) -> Self:
    """Use the scheme, client, cache dir of this cloudpath to instantiate
        a new cloudpath of the same type with the path passed.

        Used to make results of iterdir and joins have a unified client + cache.
        """
    path = str(path)
    if path.startswith('/'):
        path = path[1:]
    if not path.startswith(self.cloud_prefix):
        path = f'{self.cloud_prefix}{path}'
    return self.client.CloudPath(path)