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
@_dispatch(min=0, max=127, state=_dvistate.inpage)
def _set_char_immediate(self, char):
    self._put_char_real(char)
    self.h += self.fonts[self.f]._width_of(char)