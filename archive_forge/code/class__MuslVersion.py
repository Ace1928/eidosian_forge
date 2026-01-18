import functools
import re
import subprocess
import sys
from typing import Iterator, NamedTuple, Optional
from ._elffile import ELFFile
class _MuslVersion(NamedTuple):
    major: int
    minor: int