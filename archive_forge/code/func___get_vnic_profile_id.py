from __future__ import (absolute_import, division, print_function)
import traceback
import time
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.ovirt.ovirt.plugins.module_utils.ovirt import (
def __get_vnic_profile_id(self, nic):
    """
        Return VNIC profile ID looked up by it's name, because there can be
        more VNIC profiles with same name, other criteria of filter is cluster.
        """
    vnics_service = self._connection.system_service().vnic_profiles_service()
    clusters_service = self._connection.system_service().clusters_service()
    cluster = search_by_name(clusters_service, self.param('cluster'))
    profiles = [profile for profile in vnics_service.list() if profile.name == nic.get('profile_name')]
    cluster_networks = [net.id for net in self._connection.follow_link(cluster.networks)]
    try:
        return next((profile.id for profile in profiles if profile.network.id in cluster_networks))
    except StopIteration:
        raise Exception("Profile '%s' was not found in cluster '%s'" % (nic.get('profile_name'), self.param('cluster')))