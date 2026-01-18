from openstack.cloud import inventory
from openstack.tests.functional import base
def _cleanup_server(self):
    self.user_cloud.delete_server(self.server_id, wait=True)