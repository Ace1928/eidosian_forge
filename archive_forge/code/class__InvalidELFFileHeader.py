import collections
import functools
import os
import re
import struct
import sys
import warnings
from typing import IO, Dict, Iterator, NamedTuple, Optional, Tuple
class _InvalidELFFileHeader(ValueError):
    """
        An invalid ELF file header was found.
        """