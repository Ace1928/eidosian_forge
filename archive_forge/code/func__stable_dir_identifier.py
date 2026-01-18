from __future__ import annotations
import hashlib
import os
import time
from pathlib import Path
from streamlit.util import HASHLIB_KWARGS
def _stable_dir_identifier(dir_path: str, glob_pattern: str) -> str:
    """Wait for the files in a directory to look stable-ish before returning an id.

    We do this to deal with problems that would otherwise arise from many tools
    (e.g. git) and editors (e.g. vim) "editing" files (from the user's
    perspective) by doing some combination of deleting, creating, and moving
    various files under the hood.

    Because of this, we're unable to rely on FileSystemEvents that we receive
    from watchdog to determine when a file has been added to or removed from a
    directory.

    This is a bit of an unfortunate situation, but the approach we take here is
    most likely fine as:
      * The worst thing that can happen taking this approach is a false
        positive page added/removed notification, which isn't too disastrous
        and can just be ignored.
      * It is impossible (that is, I'm fairly certain that the problem is
        undecidable) to know whether a file created/deleted/moved event
        corresponds to a legitimate file creation/deletion/move or is part of
        some sequence of events that results in what the user sees as a file
        "edit".
    """
    dirfiles = _dirfiles(dir_path, glob_pattern)
    for _ in range(_MAX_RETRIES):
        time.sleep(_RETRY_WAIT_SECS)
        new_dirfiles = _dirfiles(dir_path, glob_pattern)
        if dirfiles == new_dirfiles:
            break
        dirfiles = new_dirfiles
    return f'{dir_path}+{dirfiles}'