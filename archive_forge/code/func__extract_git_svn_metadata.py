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
def _extract_git_svn_metadata(self, rev, message):
    lines = message.split('\n')
    if not (lines[-1] == '' and len(lines) >= 2 and lines[-2].startswith('git-svn-id:')):
        return message
    git_svn_id = lines[-2].split(': ', 1)[1]
    rev.properties['git-svn-id'] = git_svn_id
    url, rev, uuid = parse_git_svn_id(git_svn_id)
    return '\n'.join(lines[:-2])