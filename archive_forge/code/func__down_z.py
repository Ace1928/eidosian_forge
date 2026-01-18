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
@_dispatch(min=166, max=170, state=_dvistate.inpage, args=('slen',))
def _down_z(self, new_z):
    if new_z is not None:
        self.z = new_z
    self.v += self.z