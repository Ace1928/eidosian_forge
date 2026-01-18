from __future__ import annotations
import fnmatch
import os
import subprocess
import sys
import threading
import time
import typing as t
from itertools import chain
from pathlib import PurePath
from ._internal import _log
def _find_common_roots(paths: t.Iterable[str]) -> t.Iterable[str]:
    root: dict[str, dict] = {}
    for chunks in sorted((PurePath(x).parts for x in paths), key=len, reverse=True):
        node = root
        for chunk in chunks:
            node = node.setdefault(chunk, {})
        node.clear()
    rv = set()

    def _walk(node: t.Mapping[str, dict], path: tuple[str, ...]) -> None:
        for prefix, child in node.items():
            _walk(child, path + (prefix,))
        if not node:
            rv.add(os.path.join(*path))
    _walk(root, ())
    return rv