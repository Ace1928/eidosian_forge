import contextlib
import warnings
from collections import defaultdict
from enum import IntEnum
from typing import (
class ExactStr(str):
    """Class to be used in type params where no transformations are needed."""