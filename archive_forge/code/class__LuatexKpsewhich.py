from collections import namedtuple
import enum
from functools import lru_cache, partial, wraps
import logging
import os
from pathlib import Path
import re
import struct
import subprocess
import sys
import numpy as np
from matplotlib import _api, cbook
class _LuatexKpsewhich:

    @lru_cache
    def __new__(cls):
        self = object.__new__(cls)
        self._proc = self._new_proc()
        return self

    def _new_proc(self):
        return subprocess.Popen(['luatex', '--luaonly', str(cbook._get_data_path('kpsewhich.lua'))], stdin=subprocess.PIPE, stdout=subprocess.PIPE)

    def search(self, filename):
        if self._proc.poll() is not None:
            self._proc = self._new_proc()
        self._proc.stdin.write(os.fsencode(filename) + b'\n')
        self._proc.stdin.flush()
        out = self._proc.stdout.readline().rstrip()
        return None if out == b'nil' else os.fsdecode(out)