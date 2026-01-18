from __future__ import (absolute_import, division, print_function)
import traceback
import time
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.ovirt.ovirt.plugins.module_utils.ovirt import (
def _get_cluster_mappings(module):
    clusterMappings = list()
    for clusterMapping in module.params['cluster_mappings']:
        clusterMappings.append(otypes.RegistrationClusterMapping(from_=otypes.Cluster(name=clusterMapping['source_name']), to=otypes.Cluster(name=clusterMapping['dest_name']) if clusterMapping['dest_name'] else None))
    return clusterMappings