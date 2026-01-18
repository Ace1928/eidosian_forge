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
def _put_rule_real(self, a, b):
    if a > 0 and b > 0:
        self.boxes.append(Box(self.h, self.v, a, b))