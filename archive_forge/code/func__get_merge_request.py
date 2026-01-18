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
def _get_merge_request(self, project, merge_id):
    path = 'projects/%s/merge_requests/%d' % (urlutils.quote(str(project), ''), merge_id)
    response = self._api_request('GET', path)
    if response.status == 403:
        raise errors.PermissionDenied(response.text)
    if response.status != 200:
        _unexpected_status(path, response)
    return json.loads(response.data)