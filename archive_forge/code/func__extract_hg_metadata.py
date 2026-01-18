import base64
import stat
from typing import Optional
import fastbencode as bencode
from .. import errors, foreign, trace, urlutils
from ..foreign import ForeignRevision, ForeignVcs, VcsMappingRegistry
from ..revision import NULL_REVISION, Revision
from .errors import NoPushSupport
from .hg import extract_hg_metadata, format_hg_metadata
from .roundtrip import (CommitSupplement, extract_bzr_metadata,
def _extract_hg_metadata(self, rev, message):
    message, renames, branch, extra = extract_hg_metadata(message)
    if branch is not None:
        rev.properties['hg:extra:branch'] = branch
    for name, value in extra.items():
        rev.properties['hg:extra:' + name] = base64.b64encode(value)
    if renames:
        rev.properties['hg:renames'] = base64.b64encode(bencode.bencode([(new, old) for old, new in renames.items()]))
    return message