from __future__ import (absolute_import, division, print_function)
from ansible.errors import AnsibleError
from ansible.plugins.inventory import BaseInventoryPlugin, Constructable, Cacheable
from ansible.utils.unsafe_proxy import wrap_var as make_unsafe
def _get_instances_inventory(self):
    """Retrieve Linode instance information from cloud inventory."""
    try:
        self.instances = self.client.linode.instances()
    except LinodeApiError as exception:
        raise AnsibleError('Linode client raised: %s' % exception)