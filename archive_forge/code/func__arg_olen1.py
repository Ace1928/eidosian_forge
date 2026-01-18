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
def _arg_olen1(dvi, delta):
    """
    Read *delta*+1 bytes, returning the bytes interpreted as
    unsigned integer for 0<=*delta*<3 and signed if *delta*==3.
    """
    return dvi._arg(delta + 1, delta == 3)