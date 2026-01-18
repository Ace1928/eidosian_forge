import logging
from keystoneauth1 import discover
from keystoneauth1.exceptions.http import NotAcceptable
from barbicanclient import client as base_client
from barbicanclient.v1 import acls
from barbicanclient.v1 import cas
from barbicanclient.v1 import containers
from barbicanclient.v1 import orders
from barbicanclient.v1 import secrets
def _get_min_max_version(self, session, endpoint, microversion):
    try:
        resp = discover.get_version_data(session, endpoint, version_header='key-manager ' + microversion)
    except NotAcceptable:
        return (None, None)
    resp = resp[0]
    status = resp['status'].upper()
    if status == _STABLE:
        min_ver = '1.0'
        max_ver = '1.0'
    else:
        min_ver = resp['min_version']
        max_ver = resp['max_version']
    return (min_ver, max_ver)