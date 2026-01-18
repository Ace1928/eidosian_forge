import collections
import contextlib
import functools
import os
import re
import sys
import warnings
from typing import Dict, Generator, Iterator, NamedTuple, Optional, Tuple
from ._elffile import EIClass, EIData, ELFFile, EMachine
@contextlib.contextmanager
def _parse_elf(path: str) -> Generator[Optional[ELFFile], None, None]:
    try:
        with open(path, 'rb') as f:
            yield ELFFile(f)
    except (OSError, TypeError, ValueError):
        yield None