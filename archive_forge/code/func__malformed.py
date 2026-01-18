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
@_dispatch(min=250, max=255)
def _malformed(self, offset):
    raise ValueError(f'unknown command: byte {250 + offset}')