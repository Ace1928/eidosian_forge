from __future__ import annotations
from pathlib import Path
from socket import socket
from typing import Callable
from watchfiles import watch
from uvicorn.config import Config
from uvicorn.supervisors.basereload import BaseReload
class WatchFilesReload(BaseReload):

    def __init__(self, config: Config, target: Callable[[list[socket] | None], None], sockets: list[socket]) -> None:
        super().__init__(config, target, sockets)
        self.reloader_name = 'WatchFiles'
        self.reload_dirs = []
        for directory in config.reload_dirs:
            if Path.cwd() not in directory.parents:
                self.reload_dirs.append(directory)
        if Path.cwd() not in self.reload_dirs:
            self.reload_dirs.append(Path.cwd())
        self.watch_filter = FileFilter(config)
        self.watcher = watch(*self.reload_dirs, watch_filter=None, stop_event=self.should_exit, yield_on_timeout=True)

    def should_restart(self) -> list[Path] | None:
        self.pause()
        changes = next(self.watcher)
        if changes:
            unique_paths = {Path(c[1]) for c in changes}
            return [p for p in unique_paths if self.watch_filter(p)]
        return None