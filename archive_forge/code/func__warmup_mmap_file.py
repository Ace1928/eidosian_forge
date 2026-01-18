from lib2to3.pgen2 import token
import os
import torch
import numpy as np
import shutil
import struct
from functools import lru_cache
from itertools import accumulate
def _warmup_mmap_file(path):
    pass