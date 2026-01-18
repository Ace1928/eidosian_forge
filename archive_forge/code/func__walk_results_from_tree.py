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
@staticmethod
def _walk_results_from_tree(root, tree, top_down=True):
    """Utility to yield tuples in the form expected by `.walk` from the file
        tree constructed by `_build_substree`.
        """
    dirs = []
    files = []
    for item, branch in tree.items():
        files.append(item) if branch is None else dirs.append(item)
    if top_down:
        yield (root, dirs, files)
    for dir in dirs:
        yield from CloudPath._walk_results_from_tree(root / dir, tree[dir], top_down=top_down)
    if not top_down:
        yield (root, dirs, files)