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
@_dispatch(139, state=_dvistate.outer, args=('s4',) * 11)
def _bop(self, c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, p):
    self.state = _dvistate.inpage
    self.h, self.v, self.w, self.x, self.y, self.z = (0, 0, 0, 0, 0, 0)
    self.stack = []
    self.text = []
    self.boxes = []