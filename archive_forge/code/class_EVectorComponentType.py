import sys
import os
from six import iteritems
from enum import IntEnum
from contextlib import contextmanager
import json
class EVectorComponentType(IntEnum):
    Float = 0
    I8 = 1
    I32 = 2