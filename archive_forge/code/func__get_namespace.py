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
def _get_namespace(self, namespace):
    path = 'namespaces/' + urlutils.quote(str(namespace), '')
    response = self._api_request('GET', path)
    if response.status == 200:
        return json.loads(response.data)
    if response.status == 404:
        return None
    _unexpected_status(path, response)