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
def _decode_commit_message(self, rev, message, encoding):
    if rev is None:
        rev = Revision()
    message = self._extract_hg_metadata(rev, message)
    message = self._extract_git_svn_metadata(rev, message)
    message, metadata = self._extract_bzr_metadata(rev, message)
    try:
        return (message.decode(encoding), metadata)
    except LookupError:
        raise UnknownCommitEncoding(encoding)