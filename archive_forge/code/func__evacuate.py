import base64
import collections
from urllib import parse
from novaclient import api_versions
from novaclient import base
from novaclient import crypto
from novaclient import exceptions
from novaclient.i18n import _
def _evacuate(self, server, host, on_shared_storage, password, force):
    """Inner function to abstract changes in evacuate API."""
    body = {}
    if on_shared_storage is not None:
        body['onSharedStorage'] = on_shared_storage
    if host is not None:
        body['host'] = host
    if password is not None:
        body['adminPass'] = password
    if force:
        body['force'] = force
    resp, body = self._action_return_resp_and_body('evacuate', server, body)
    return base.TupleWithMeta((resp, body), resp)