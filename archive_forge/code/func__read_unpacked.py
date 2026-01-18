import contextlib
import functools
import operator
import os
import re
import struct
import subprocess
import sys
from typing import IO, Iterator, NamedTuple, Optional, Tuple
def _read_unpacked(f: IO[bytes], fmt: str) -> Tuple[int, ...]:
    return struct.unpack(fmt, f.read(struct.calcsize(fmt)))