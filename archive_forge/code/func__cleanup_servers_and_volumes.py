import datetime
from fixtures import TimeoutException
from openstack import exceptions
from openstack.tests.functional import base
from openstack import utils
def _cleanup_servers_and_volumes(self, server_name):
    """Delete the named server and any attached volumes.

        Adding separate cleanup calls for servers and volumes can be tricky
        since they need to be done in the proper order. And sometimes deleting
        a server can start the process of deleting a volume if it is booted
        from that volume. This encapsulates that logic.
        """
    server = self.user_cloud.get_server(server_name)
    if not server:
        return
    volumes = self.user_cloud.get_volumes(server)
    try:
        self.user_cloud.delete_server(server.name, wait=True)
        for volume in volumes:
            if volume.status != 'deleting':
                self.user_cloud.delete_volume(volume.id, wait=True)
    except (exceptions.ResourceTimeout, TimeoutException):
        self.user_cloud.delete_server(server.name)
        for volume in volumes:
            self.operator_cloud.delete_volume(volume.id, wait=False, force=True)