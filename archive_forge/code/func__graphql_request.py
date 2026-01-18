import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional
from ... import bedding
from ... import branch as _mod_branch
from ... import controldir, errors, hooks, urlutils
from ... import version_string as breezy_version
from ...config import AuthenticationConfig, GlobalStack
from ...errors import (InvalidHttpResponse, PermissionDenied,
from ...forge import (Forge, ForgeLoginRequired, MergeProposal,
from ...git.urls import git_url_to_bzr_url
from ...i18n import gettext
from ...trace import note
from ...transport import get_transport
from ...transport.http import default_user_agent
def _graphql_request(self, body, **kwargs):
    headers = {}
    if self._token:
        headers['Authorization'] = 'token %s' % self._token
    url = urlutils.join(self.transport.base, 'graphql')
    response = self.transport.request('POST', url, headers=headers, body=json.dumps({'query': body, 'variables': kwargs}).encode('utf-8'))
    if response.status != 200:
        raise UnexpectedHttpStatus(url, response.status, headers=response.getheaders())
    data = json.loads(response.text)
    if data.get('errors'):
        raise GraphqlErrors(data.get('errors'))
    return data['data']