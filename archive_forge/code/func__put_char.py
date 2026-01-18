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
@_dispatch(min=133, max=136, state=_dvistate.inpage, args=('olen1',))
def _put_char(self, char):
    self._put_char_real(char)