from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import json
import re
import threading
from containerregistry.client import docker_creds
from containerregistry.client import docker_name
from containerregistry.client.v2_2 import docker_creds as v2_2_creds
import httplib2
import six.moves.http_client
import six.moves.urllib.parse
def _Ping(self):
    """Ping the v2 Registry.

    Only called during transport construction, this pings the listed
    v2 registry.  The point of this ping is to establish the "realm"
    and "service" to use for Basic for Bearer-Token exchanges.
    """
    headers = {'content-type': 'application/json', 'user-agent': docker_name.USER_AGENT}
    resp, content = self._transport.request('{scheme}://{registry}/v2/'.format(scheme=Scheme(self._name.registry), registry=self._name.registry), 'GET', body=None, headers=headers)
    _CheckState(resp.status in [six.moves.http_client.OK, six.moves.http_client.UNAUTHORIZED], 'Unexpected response pinging the registry: {}\nBody: {}'.format(resp.status, content or '<empty>'))
    if resp.status == six.moves.http_client.OK:
        self._authentication = _ANONYMOUS
        self._service = 'none'
        self._realm = 'none'
        return
    challenge = resp['www-authenticate']
    _CheckState(' ' in challenge, 'Unexpected "www-authenticate" header form: %s' % challenge)
    self._authentication, remainder = challenge.split(' ', 1)
    self._authentication = self._authentication.capitalize()
    _CheckState(self._authentication in [_BASIC, _BEARER], 'Unexpected "www-authenticate" challenge type: %s' % self._authentication)
    self._service = self._name.registry
    tokens = remainder.split(',')
    for t in tokens:
        if t.startswith(_REALM_PFX):
            self._realm = t[len(_REALM_PFX):].strip('"')
        elif t.startswith(_SERVICE_PFX):
            self._service = t[len(_SERVICE_PFX):].strip('"')
    _CheckState(self._realm, 'Expected a "%s" in "www-authenticate" header: %s' % (_REALM_PFX, challenge))