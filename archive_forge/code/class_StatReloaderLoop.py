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
class StatReloaderLoop(ReloaderLoop):
    name = 'stat'

    def __enter__(self) -> ReloaderLoop:
        self.mtimes: dict[str, float] = {}
        return super().__enter__()

    def run_step(self) -> None:
        for name in _find_stat_paths(self.extra_files, self.exclude_patterns):
            try:
                mtime = os.stat(name).st_mtime
            except OSError:
                continue
            old_time = self.mtimes.get(name)
            if old_time is None:
                self.mtimes[name] = mtime
                continue
            if mtime > old_time:
                self.trigger_reload(name)