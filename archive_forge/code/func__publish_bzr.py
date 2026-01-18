import re
import shutil
import tempfile
from typing import Any, List, Optional
from ... import branch as _mod_branch
from ... import controldir, errors, hooks, urlutils
from ...forge import (AutoMergeUnsupported, Forge, LabelsUnsupported,
from ...git.urls import git_url_to_bzr_url
from ...lazy_import import lazy_import
from ...trace import mutter
from breezy.plugins.launchpad import (
from ...transport import get_transport
def _publish_bzr(self, local_branch, base_branch, name, owner, project=None, revision_id=None, overwrite=False, allow_lossy=True, tag_selector=None):
    to_path = self._get_derived_bzr_path(base_branch, name, owner, project)
    to_transport = get_transport(BZR_SCHEME_MAP['ssh'] + to_path)
    try:
        dir_to = controldir.ControlDir.open_from_transport(to_transport)
    except errors.NotBranchError:
        dir_to = None
    if dir_to is None:
        br_to = local_branch.create_clone_on_transport(to_transport, revision_id=revision_id, tag_selector=tag_selector)
    else:
        br_to = dir_to.push_branch(local_branch, revision_id, overwrite=overwrite, tag_selector=tag_selector).target_branch
    return (br_to, 'https://code.launchpad.net/' + to_path)