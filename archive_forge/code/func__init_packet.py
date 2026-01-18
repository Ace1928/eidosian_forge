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
def _init_packet(self, pl):
    if self.state != _dvistate.outer:
        raise ValueError('Misplaced packet in vf file')
    self.h, self.v, self.w, self.x, self.y, self.z = (0, 0, 0, 0, 0, 0)
    self.stack, self.text, self.boxes = ([], [], [])
    self.f = self._first_font
    return self.file.tell() + pl