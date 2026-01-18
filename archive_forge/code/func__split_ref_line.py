import os
import warnings
from contextlib import suppress
from typing import Any, Dict, Optional, Set
from .errors import PackedRefsException, RefFormatError
from .file import GitFile, ensure_dir_exists
from .objects import ZERO_SHA, ObjectID, Tag, git_line, valid_hexsha
from .pack import ObjectContainer
def _split_ref_line(line):
    """Split a single ref line into a tuple of SHA1 and name."""
    fields = line.rstrip(b'\n\r').split(b' ')
    if len(fields) != 2:
        raise PackedRefsException('invalid ref line %r' % line)
    sha, name = fields
    if not valid_hexsha(sha):
        raise PackedRefsException('Invalid hex sha %r' % sha)
    if not check_ref_format(name):
        raise PackedRefsException('invalid ref name %r' % name)
    return (sha, name)