from lib2to3.pgen2 import token
import os
import torch
import numpy as np
import shutil
import struct
from functools import lru_cache
from itertools import accumulate
@staticmethod
def _get_pointers(sizes):
    dtype_size = dtype().itemsize
    address = 0
    pointers = []
    for size in sizes:
        pointers.append(address)
        address += size * dtype_size
    return pointers