import json
import os
import re
import time
from datetime import datetime
from typing import Optional
from ... import bedding
from ... import branch as _mod_branch
from ... import controldir, errors, urlutils
from ...forge import (Forge, ForgeLoginRequired, MergeProposal,
from ...git.urls import git_url_to_bzr_url
from ...trace import mutter
from ...transport import get_transport
def _branch_url_from_project(self, project_id, branch_name, *, preferred_schemes=None):
    if project_id is None:
        return None
    project = self.gl._get_project(project_id)
    if preferred_schemes is None:
        preferred_schemes = DEFAULT_PREFERRED_SCHEMES
    for scheme in preferred_schemes:
        if scheme in SCHEME_MAP:
            return gitlab_url_to_bzr_url(project[SCHEME_MAP[scheme]], branch_name)
    raise KeyError