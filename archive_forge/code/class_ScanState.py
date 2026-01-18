import collections
import enum
import dataclasses
import itertools as it
import os
import pickle
import re
import shutil
import subprocess
import sys
import textwrap
from typing import (
import torch
from torch.utils.benchmark.utils import common, cpp_jit
from torch.utils.benchmark.utils._stubs import CallgrindModuleType
class ScanState(enum.Enum):
    SCANNING_FOR_TOTAL = 0
    SCANNING_FOR_START = 1
    PARSING = 2