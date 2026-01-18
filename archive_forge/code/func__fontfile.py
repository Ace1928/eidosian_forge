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
@lru_cache
def _fontfile(cls, suffix, texname):
    return cls(find_tex_file(texname + suffix))