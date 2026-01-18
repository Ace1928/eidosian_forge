from __future__ import annotations
import subprocess as S
from threading import Thread
import typing as T
import re
import os
from .. import mlog
from ..mesonlib import PerMachine, Popen_safe, version_compare, is_windows, OptionKey
from ..programs import find_external_program, NonExistingExternalProgram
def _call_cmout(self, args: T.List[str], build_dir: Path, env: T.Optional[T.Dict[str, str]]) -> TYPE_result:
    cmd = self.cmakebin.get_command() + args
    proc = S.Popen(cmd, stdout=S.PIPE, stderr=S.STDOUT, cwd=str(build_dir), env=env)
    while True:
        line = proc.stdout.readline()
        if not line:
            break
        mlog.log(line.decode(errors='ignore').strip('\n'))
    proc.stdout.close()
    proc.wait()
    return (proc.returncode, None, None)