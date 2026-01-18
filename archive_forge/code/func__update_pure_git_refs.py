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
def _update_pure_git_refs(result, new_refs, overwrite, tag_selector, old_refs):
    mutter('updating refs. old refs: %r, new refs: %r', old_refs, new_refs)
    result.tag_updates = {}
    result.tag_conflicts = []
    ret = {}

    def ref_equals(refs, name, git_sha, revid):
        try:
            value = refs[name]
        except KeyError:
            return False
        if value[0] is not None and git_sha is not None:
            return value[0] == git_sha
        if value[1] is not None and revid is not None:
            return value[1] == revid
        return False
    for ref, (git_sha, revid) in new_refs.items():
        if ref_equals(ret, ref, git_sha, revid):
            if git_sha is None:
                git_sha = old_refs[ref][0]
            if revid is None:
                revid = old_refs[ref][1]
            ret[ref] = new_refs[ref] = (git_sha, revid)
        elif ref not in ret or overwrite:
            try:
                tag_name = ref_to_tag_name(ref)
            except ValueError:
                pass
            else:
                if tag_selector and (not tag_selector(tag_name)):
                    continue
                result.tag_updates[tag_name] = revid
            ret[ref] = (git_sha, revid)
        else:
            diverged = False
            if diverged:
                try:
                    name = ref_to_tag_name(ref)
                except ValueError:
                    pass
                else:
                    result.tag_conflicts.append((name, revid, ret[name][1]))
            else:
                ret[ref] = (git_sha, revid)
    return ret