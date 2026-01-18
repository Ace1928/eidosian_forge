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
def _merge_mr(self, project_id, iid, kwargs):
    path = 'projects/{}/merge_requests/{}/merge'.format(urlutils.quote(str(project_id), ''), iid)
    response = self._api_request('PUT', path, fields=kwargs)
    if response.status == 200:
        return json.loads(response.data)
    if response.status == 403:
        raise errors.PermissionDenied(response.text)
    _unexpected_status(path, response)