import array
import enum
import functools
import logging
import struct
import sys
from typing import List, NamedTuple, Optional, Tuple
import torch
class OperandValueSourceType:
    IMMEDIATE = 0
    NUMBERED_BUFFER = 2
    NUMBERED_MEMORY = 3