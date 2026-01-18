from __future__ import annotations
import logging
import warnings
from pathlib import Path
from socket import socket
from typing import TYPE_CHECKING, Callable
from watchgod import DefaultWatcher
from uvicorn.config import Config
from uvicorn.supervisors.basereload import BaseReload
class WatchGodReload(BaseReload):

    def __init__(self, config: Config, target: Callable[[list[socket] | None], None], sockets: list[socket]) -> None:
        warnings.warn('"watchgod" is deprecated, you should switch to watchfiles (`pip install watchfiles`).', DeprecationWarning)
        super().__init__(config, target, sockets)
        self.reloader_name = 'WatchGod'
        self.watchers = []
        reload_dirs = []
        for directory in config.reload_dirs:
            if Path.cwd() not in directory.parents:
                reload_dirs.append(directory)
        if Path.cwd() not in reload_dirs:
            reload_dirs.append(Path.cwd())
        for w in reload_dirs:
            self.watchers.append(CustomWatcher(w.resolve(), self.config))

    def should_restart(self) -> list[Path] | None:
        self.pause()
        for watcher in self.watchers:
            change = watcher.check()
            if change != set():
                return list({Path(c[1]) for c in change})
        return None