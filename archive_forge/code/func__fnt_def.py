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
@_dispatch(min=243, max=246, args=('olen1', 'u4', 'u4', 'u4', 'u1', 'u1'))
def _fnt_def(self, k, c, s, d, a, l):
    self._fnt_def_real(k, c, s, d, a, l)