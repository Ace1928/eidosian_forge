import re
from git.cmd import handle_process_output
from git.compat import defenc
from git.util import finalize_process, hex_to_bin
from .objects.blob import Blob
from .objects.util import mode_str_to_int
from typing import (
from git.types import PathLike, Literal
def _octal_repl(matchobj: Match) -> bytes:
    value = matchobj.group(1)
    value = int(value, 8)
    value = bytes(bytearray((value,)))
    return value