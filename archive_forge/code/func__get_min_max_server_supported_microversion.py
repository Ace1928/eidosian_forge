import logging
from keystoneauth1 import discover
from keystoneauth1.exceptions.http import NotAcceptable
from barbicanclient import client as base_client
from barbicanclient.v1 import acls
from barbicanclient.v1 import cas
from barbicanclient.v1 import containers
from barbicanclient.v1 import orders
from barbicanclient.v1 import secrets
def _get_min_max_server_supported_microversion(self, session, endpoint, version, service_type, service_name, interface, region_name):
    if not endpoint:
        endpoint = session.get_endpoint(service_type=service_type, service_name=service_name, interface=interface, region_name=region_name, version=version)
    return self._get_min_max_version(session, endpoint, '1.1')