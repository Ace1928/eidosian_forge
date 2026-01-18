from __future__ import annotations
import subprocess as S
from threading import Thread
import typing as T
import re
import os
from .. import mlog
from ..mesonlib import PerMachine, Popen_safe, version_compare, is_windows, OptionKey
from ..programs import find_external_program, NonExistingExternalProgram
def _call_quiet(self, args: T.List[str], build_dir: Path, env: T.Optional[T.Dict[str, str]]) -> TYPE_result:
    build_dir.mkdir(parents=True, exist_ok=True)
    cmd = self.cmakebin.get_command() + args
    ret = S.run(cmd, env=env, cwd=str(build_dir), close_fds=False, stdout=S.PIPE, stderr=S.PIPE, universal_newlines=False)
    rc = ret.returncode
    out = ret.stdout.decode(errors='ignore')
    err = ret.stderr.decode(errors='ignore')
    return (rc, out, err)