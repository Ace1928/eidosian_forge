import os
from heat.common.i18n import _
from heatclient import client as heat_client
from keystoneauth1.identity.generic import password
from keystoneauth1 import session
from keystoneclient.v3 import client as kc_v3
from novaclient import client as nova_client
from swiftclient import client as swift_client
def _get_orchestration_client(self):
    endpoint = os.environ.get('HEAT_URL')
    if os.environ.get('OS_NO_CLIENT_AUTH') == 'True':
        session = None
    else:
        session = self.identity_client.session
    return heat_client.Client(self.HEATCLIENT_VERSION, endpoint, session=session, endpoint_type='publicURL', service_type='orchestration', region_name=self.conf.region, username=self._username(), password=self._password())