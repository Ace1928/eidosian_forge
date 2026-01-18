import functools
import re
import subprocess
import sys
from typing import Iterator, NamedTuple, Optional
from ._elffile import ELFFile
@functools.lru_cache()
def _get_musl_version(executable: str) -> Optional[_MuslVersion]:
    """Detect currently-running musl runtime version.

    This is done by checking the specified executable's dynamic linking
    information, and invoking the loader to parse its output for a version
    string. If the loader is musl, the output would be something like::

        musl libc (x86_64)
        Version 1.2.2
        Dynamic Program Loader
    """
    try:
        with open(executable, 'rb') as f:
            ld = ELFFile(f).interpreter
    except (OSError, TypeError, ValueError):
        return None
    if ld is None or 'musl' not in ld:
        return None
    proc = subprocess.run([ld], stderr=subprocess.PIPE, universal_newlines=True)
    return _parse_musl_version(proc.stderr)