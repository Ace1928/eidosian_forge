from __future__ import annotations
from pathlib import Path
from socket import socket
from typing import Callable
from watchfiles import watch
from uvicorn.config import Config
from uvicorn.supervisors.basereload import BaseReload
class FileFilter:

    def __init__(self, config: Config):
        default_includes = ['*.py']
        self.includes = [default for default in default_includes if default not in config.reload_excludes]
        self.includes.extend(config.reload_includes)
        self.includes = list(set(self.includes))
        default_excludes = ['.*', '.py[cod]', '.sw.*', '~*']
        self.excludes = [default for default in default_excludes if default not in config.reload_includes]
        self.exclude_dirs = []
        for e in config.reload_excludes:
            p = Path(e)
            try:
                is_dir = p.is_dir()
            except OSError:
                is_dir = False
            if is_dir:
                self.exclude_dirs.append(p)
            else:
                self.excludes.append(e)
        self.excludes = list(set(self.excludes))

    def __call__(self, path: Path) -> bool:
        for include_pattern in self.includes:
            if path.match(include_pattern):
                if str(path).endswith(include_pattern):
                    return True
                for exclude_dir in self.exclude_dirs:
                    if exclude_dir in path.parents:
                        return False
                for exclude_pattern in self.excludes:
                    if path.match(exclude_pattern):
                        return False
                return True
        return False