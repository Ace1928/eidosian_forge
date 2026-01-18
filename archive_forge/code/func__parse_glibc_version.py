import collections
import contextlib
import functools
import os
import re
import sys
import warnings
from typing import Dict, Generator, Iterator, NamedTuple, Optional, Tuple
from ._elffile import EIClass, EIData, ELFFile, EMachine
def _parse_glibc_version(version_str: str) -> Tuple[int, int]:
    """Parse glibc version.

    We use a regexp instead of str.split because we want to discard any
    random junk that might come after the minor version -- this might happen
    in patched/forked versions of glibc (e.g. Linaro's version of glibc
    uses version strings like "2.20-2014.11"). See gh-3588.
    """
    m = re.match('(?P<major>[0-9]+)\\.(?P<minor>[0-9]+)', version_str)
    if not m:
        warnings.warn(f'Expected glibc version with 2 components major.minor, got: {version_str}', RuntimeWarning)
        return (-1, -1)
    return (int(m.group('major')), int(m.group('minor')))