import os
import warnings
from contextlib import suppress
from typing import Any, Dict, Optional, Set
from .errors import PackedRefsException, RefFormatError
from .file import GitFile, ensure_dir_exists
from .objects import ZERO_SHA, ObjectID, Tag, git_line, valid_hexsha
from .pack import ObjectContainer
def _set_default_branch(refs: RefsContainer, origin: bytes, origin_head: bytes, branch: bytes, ref_message: Optional[bytes]) -> bytes:
    """Set the default branch."""
    origin_base = b'refs/remotes/' + origin + b'/'
    if branch:
        origin_ref = origin_base + branch
        if origin_ref in refs:
            local_ref = LOCAL_BRANCH_PREFIX + branch
            refs.add_if_new(local_ref, refs[origin_ref], ref_message)
            head_ref = local_ref
        elif LOCAL_TAG_PREFIX + branch in refs:
            head_ref = LOCAL_TAG_PREFIX + branch
        else:
            raise ValueError('%r is not a valid branch or tag' % os.fsencode(branch))
    elif origin_head:
        head_ref = origin_head
        if origin_head.startswith(LOCAL_BRANCH_PREFIX):
            origin_ref = origin_base + origin_head[len(LOCAL_BRANCH_PREFIX):]
        else:
            origin_ref = origin_head
        try:
            refs.add_if_new(head_ref, refs[origin_ref], ref_message)
        except KeyError:
            pass
    else:
        raise ValueError('neither origin_head nor branch are provided')
    return head_ref