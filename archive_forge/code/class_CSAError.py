import numpy as np
from .structreader import Unpacker
from .utils import find_private_section
class CSAError(Exception):
    pass