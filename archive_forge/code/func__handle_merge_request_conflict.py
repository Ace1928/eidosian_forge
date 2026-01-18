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
def _handle_merge_request_conflict(self, message, source_url, target_project):
    m = re.fullmatch('Another open merge request already exists for this source branch: \\!([0-9]+)', message[0])
    if m:
        merge_id = int(m.group(1))
        mr = self._get_merge_request(target_project, merge_id)
        raise MergeProposalExists(source_url, GitLabMergeProposal(self, mr))
    raise MergeRequestConflict(message)