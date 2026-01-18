from enum import IntFlag, auto
from typing import Dict, Tuple
from ._utils import deprecate_with_replacement
class TrailerKeys:
    ROOT = '/Root'
    ENCRYPT = '/Encrypt'
    ID = '/ID'
    INFO = '/Info'
    SIZE = '/Size'