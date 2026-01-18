import os
import warnings
from contextlib import suppress
from typing import Any, Dict, Optional, Set
from .errors import PackedRefsException, RefFormatError
from .file import GitFile, ensure_dir_exists
from .objects import ZERO_SHA, ObjectID, Tag, git_line, valid_hexsha
from .pack import ObjectContainer
def _set_origin_head(refs, origin, origin_head):
    origin_base = b'refs/remotes/' + origin + b'/'
    if origin_head and origin_head.startswith(LOCAL_BRANCH_PREFIX):
        origin_ref = origin_base + HEADREF
        target_ref = origin_base + origin_head[len(LOCAL_BRANCH_PREFIX):]
        if target_ref in refs:
            refs.set_symbolic_ref(origin_ref, target_ref)