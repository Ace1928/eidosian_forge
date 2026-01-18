import re
from git.cmd import handle_process_output
from git.compat import defenc
from git.util import finalize_process, hex_to_bin
from .objects.blob import Blob
from .objects.util import mode_str_to_int
from typing import (
from git.types import PathLike, Literal
@classmethod
def _pick_best_path(cls, path_match: bytes, rename_match: bytes, path_fallback_match: bytes) -> Optional[bytes]:
    if path_match:
        return decode_path(path_match)
    if rename_match:
        return decode_path(rename_match, has_ab_prefix=False)
    if path_fallback_match:
        return decode_path(path_fallback_match)
    return None