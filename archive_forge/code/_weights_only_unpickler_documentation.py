import functools as _functools
from collections import OrderedDict
from pickle import (
from struct import unpack
from sys import maxsize
from typing import Any, Dict, List
import torch
Read a pickled object representation from the open file.

        Return the reconstituted object hierarchy specified in the file.
        