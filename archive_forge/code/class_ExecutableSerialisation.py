from __future__ import annotations
from dataclasses import dataclass
import os
import abc
import typing as T
@dataclass(eq=False)
class ExecutableSerialisation:
    cmd_args: T.List[str]
    env: T.Optional[EnvironmentVariables] = None
    exe_wrapper: T.Optional['programs.ExternalProgram'] = None
    workdir: T.Optional[str] = None
    extra_paths: T.Optional[T.List] = None
    capture: T.Optional[str] = None
    feed: T.Optional[str] = None
    tag: T.Optional[str] = None
    verbose: bool = False
    installdir_map: T.Optional[T.Dict[str, str]] = None

    def __post_init__(self) -> None:
        self.pickled = False
        self.skip_if_destdir = False
        self.subproject = ''
        self.dry_run = False