import os
import numbers
from pathlib import Path
from typing import Union, Set
import numpy as np
from ase.io.jsonio import encode, decode
from ase.utils import plural
class InvalidULMFileError(IOError):
    pass