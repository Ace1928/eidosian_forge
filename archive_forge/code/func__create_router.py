import uuid
from openstackclient.tests.functional.network.v2 import common
def _create_router(self):
    router_name = uuid.uuid4().hex
    json_output = self.openstack('router create ' + router_name, parse_output=True)
    self.assertIsNotNone(json_output['id'])
    router_id = json_output['id']
    self.addCleanup(self.openstack, 'router delete ' + router_id)
    return router_id