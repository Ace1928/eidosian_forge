from __future__ import (absolute_import, division, print_function)
from ansible.utils.sentinel import Sentinel
class ConnectionFieldAttribute(FieldAttribute):

    def __get__(self, obj, obj_type=None):
        from ansible.module_utils.compat.paramiko import paramiko
        from ansible.utils.ssh_functions import check_for_controlpersist
        value = super().__get__(obj, obj_type)
        if value == 'smart':
            value = 'ssh'
            if not check_for_controlpersist('ssh') and paramiko is not None:
                value = 'paramiko'
        elif value == 'persistent' and paramiko is not None:
            value = 'paramiko'
        return value