from __future__ import (absolute_import, division, print_function)
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.ovirt.ovirt.plugins.module_utils.ovirt import (
def _objects_service(connection, object_type):
    if object_type == 'system':
        return connection.system_service()
    return getattr(connection.system_service(), '%ss_service' % object_type, None)()