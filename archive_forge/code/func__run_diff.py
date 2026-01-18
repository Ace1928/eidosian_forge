from __future__ import annotations
from argparse import ArgumentParser
from argparse import Namespace
import contextlib
import difflib
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
from typing import Any
from typing import Dict
from typing import Iterator
from typing import Optional
from typing import Union
from . import compat
def _run_diff(self, destination_path: Union[str, Path], *, source: Optional[str]=None, source_file: Optional[str]=None) -> None:
    if source_file:
        with open(source_file, encoding='utf-8') as tf:
            source_lines = list(tf)
    elif source is not None:
        source_lines = source.splitlines(keepends=True)
    else:
        assert False, 'source or source_file is required'
    with open(destination_path, encoding='utf-8') as dp:
        d = difflib.unified_diff(list(dp), source_lines, fromfile=Path(destination_path).as_posix(), tofile='<proposed changes>', n=3, lineterm='\n')
        d_as_list = list(d)
        if d_as_list:
            self.diffs_detected = True
            print(''.join(d_as_list))