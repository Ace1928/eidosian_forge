import collections
from unittest import mock
import uuid
from openstack.network.v2 import vpn_endpoint_group as vpn_epg
from openstack.network.v2 import vpn_ike_policy as vpn_ikep
from openstack.network.v2 import vpn_ipsec_policy as vpn_ipsecp
from openstack.network.v2 import vpn_ipsec_site_connection as vpn_sitec
from openstack.network.v2 import vpn_service
class IPSecPolicy(FakeVPNaaS):
    """Fake one or more IPsec policies"""

    def __init__(self):
        super(IPSecPolicy, self).__init__()
        self.ordered = collections.OrderedDict((('id', 'ikepolicy-id-' + uuid.uuid4().hex), ('name', 'my-ikepolicy-' + uuid.uuid4().hex), ('auth_algorithm', 'sha1'), ('encapsulation_mode', 'tunnel'), ('transform_protocol', 'esp'), ('encryption_algorithm', 'aes-128'), ('pfs', 'group5'), ('description', 'my-desc-' + uuid.uuid4().hex), ('project_id', 'project-id-' + uuid.uuid4().hex), ('lifetime', {'units': 'seconds', 'value': 3600})))