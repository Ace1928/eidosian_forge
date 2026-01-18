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
@_dispatch(137, state=_dvistate.inpage, args=('s4', 's4'))
def _put_rule(self, a, b):
    self._put_rule_real(a, b)