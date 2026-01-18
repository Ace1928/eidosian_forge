from ansible_collections.openstack.cloud.plugins.module_utils.openstack import OpenStackModule
def _build_pool(self):
    pool_start = self.params['allocation_pool_start']
    pool_end = self.params['allocation_pool_end']
    if pool_start:
        return [dict(start=pool_start, end=pool_end)]
    return None