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
def _list_merge_requests(self, author=None, project=None, state=None):
    if project is not None:
        path = 'projects/%s/merge_requests' % urlutils.quote(str(project), '')
    else:
        path = 'merge_requests'
    parameters = {}
    if state:
        parameters['state'] = state
    if author:
        parameters['author_username'] = urlutils.quote(author, '')
    return self._list_paged(path, parameters, per_page=DEFAULT_PAGE_SIZE)