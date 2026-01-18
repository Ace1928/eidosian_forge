import os
from heat.common.i18n import _
from heatclient import client as heat_client
from keystoneauth1.identity.generic import password
from keystoneauth1 import session
from keystoneclient.v3 import client as kc_v3
from novaclient import client as nova_client
from swiftclient import client as swift_client
def _get_object_client(self):
    args = {'auth_version': self.auth_version, 'session': self.identity_client.session, 'os_options': {'endpoint_type': 'publicURL', 'region_name': self.conf.region, 'service_type': 'object-store'}}
    return swift_client.Connection(**args)