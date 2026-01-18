import contextlib
from collections import defaultdict
from functools import partial
from io import BytesIO
from typing import Dict, Optional, Set, Tuple
from dulwich.config import ConfigFile as GitConfigFile
from dulwich.config import parse_submodules
from dulwich.objects import ZERO_SHA, NotCommitError
from dulwich.repo import check_ref_format
from .. import branch, config, controldir, errors, lock
from .. import repository as _mod_repository
from .. import revision, trace, transport, urlutils
from ..foreign import ForeignBranch
from ..revision import NULL_REVISION
from ..tag import InterTags, TagConflict, Tags, TagSelector, TagUpdates
from ..trace import is_quiet, mutter, warning
from .errors import NoPushSupport
from .mapping import decode_git_path, encode_git_path
from .push import remote_divergence
from .refs import (branch_name_to_ref, is_tag, ref_to_branch_name,
from .unpeel_map import UnpeelMap
from .urls import bzr_url_to_git_url, git_url_to_bzr_url
def _get_new_refs(self, stop_revision=None, fetch_tags=None, stop_revno=None):
    if not self.source.is_locked():
        raise errors.ObjectNotLocked(self.source)
    if stop_revision is None:
        stop_revno, stop_revision = self.source.last_revision_info()
    elif stop_revno is None:
        try:
            stop_revno = self.source.revision_id_to_revno(stop_revision)
        except errors.NoSuchRevision:
            stop_revno = None
    if not isinstance(stop_revision, bytes):
        raise TypeError(stop_revision)
    main_ref = self.target.ref
    refs = {main_ref: (None, stop_revision)}
    if fetch_tags is None:
        c = self.source.get_config_stack()
        fetch_tags = c.get('branch.fetch_tags')
    for name, revid in self.source.tags.get_tag_dict().items():
        if self.source.repository.has_revision(revid):
            ref = tag_name_to_ref(name)
            if not check_ref_format(ref):
                warning('skipping tag with invalid characters %s (%s)', name, ref)
                continue
            if fetch_tags:
                refs[ref] = (None, revid)
    return (refs, main_ref, (stop_revno, stop_revision))