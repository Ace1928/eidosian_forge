import os
import numpy as np
import threading
from time import time
from .. import config, logging
def _use_gb_ram(mem_gb):
    """A test function to consume mem_gb GB of RAM"""
    num_bytes = int(mem_gb * _GB)
    gb_str = ' ' * ((num_bytes - BOFFSET) // BSIZE)
    assert sys.getsizeof(gb_str) == num_bytes
    return gb_str