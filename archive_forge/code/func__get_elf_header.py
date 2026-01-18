import collections
import functools
import os
import re
import struct
import sys
import warnings
from typing import IO, Dict, Iterator, NamedTuple, Optional, Tuple
def _get_elf_header() -> Optional[_ELFFileHeader]:
    try:
        with open(sys.executable, 'rb') as f:
            elf_header = _ELFFileHeader(f)
    except (OSError, TypeError, _ELFFileHeader._InvalidELFFileHeader):
        return None
    return elf_header