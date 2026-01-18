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
def _new_proc(self):
    return subprocess.Popen(['luatex', '--luaonly', str(cbook._get_data_path('kpsewhich.lua'))], stdin=subprocess.PIPE, stdout=subprocess.PIPE)