import uuid
from openstackclient.tests.functional.network.v2 import common
def _create_helpers(self, router_id, helpers):
    created_helpers = []
    for helper in helpers:
        output = self.openstack('network l3 conntrack helper create %(router)s --helper %(helper)s --protocol %(protocol)s --port %(port)s ' % {'router': router_id, 'helper': helper['helper'], 'protocol': helper['protocol'], 'port': helper['port']}, parse_output=True)
        self.assertEqual(helper['helper'], output['helper'])
        self.assertEqual(helper['protocol'], output['protocol'])
        self.assertEqual(helper['port'], output['port'])
        created_helpers.append(output)
    return created_helpers