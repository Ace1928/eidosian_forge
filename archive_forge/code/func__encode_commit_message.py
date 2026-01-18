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
def _encode_commit_message(self, rev, message, encoding):
    ret = message.encode(encoding)
    ret += self._generate_hg_message_tail(rev)
    ret += self._generate_git_svn_metadata(rev, encoding)
    return ret