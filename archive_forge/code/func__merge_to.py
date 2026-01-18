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
def _merge_to(self, to_tags, source_tag_refs, overwrite=False, selector=None, ignore_master=False):
    unpeeled_map = defaultdict(set)
    conflicts = []
    updates = {}
    result = dict(to_tags.get_tag_dict())
    for ref_name, tag_name, peeled, unpeeled in source_tag_refs:
        if selector and (not selector(tag_name)):
            continue
        if unpeeled is not None:
            unpeeled_map[peeled].add(unpeeled)
        try:
            bzr_revid = self.source.branch.lookup_foreign_revision_id(peeled)
        except NotCommitError:
            continue
        if result.get(tag_name) == bzr_revid:
            pass
        elif tag_name not in result or overwrite:
            result[tag_name] = bzr_revid
            updates[tag_name] = bzr_revid
        else:
            conflicts.append((tag_name, bzr_revid, result[tag_name]))
    to_tags._set_tag_dict(result)
    if len(unpeeled_map) > 0:
        map_file = UnpeelMap.from_repository(to_tags.branch.repository)
        map_file.update(unpeeled_map)
        map_file.save_in_repository(to_tags.branch.repository)
    return (updates, set(conflicts))