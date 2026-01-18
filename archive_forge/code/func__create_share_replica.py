from manilaclient import api_versions
from manilaclient import base
from manilaclient.common import constants
def _create_share_replica(self, share, availability_zone=None, scheduler_hints=None, share_network=None):
    """Create a replica for a share.

        :param share: The share to create the replica of. Can be the share
        object or its UUID.
        :param availability_zone: The 'availability_zone' object or its UUID.
        :param scheduler_hints: The scheduler_hints as key=value pair. Only
        supported key is 'only_host'.
        :param share_network: either share network object or its UUID.
        """
    share_id = base.getid(share)
    body = {'share_id': share_id}
    if availability_zone:
        body['availability_zone'] = base.getid(availability_zone)
    if scheduler_hints:
        body['scheduler_hints'] = scheduler_hints
    if share_network:
        body['share_network_id'] = base.getid(share_network)
    return self._create(RESOURCES_PATH, {RESOURCE_NAME: body}, RESOURCE_NAME)