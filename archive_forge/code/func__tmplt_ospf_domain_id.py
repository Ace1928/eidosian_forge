from __future__ import absolute_import, division, print_function
import re
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.rm_base.network_template import (
def _tmplt_ospf_domain_id(config_data):
    if 'domain_id' in config_data:
        command = 'domain-id'
        if 'ip_address' in config_data['domain_id']:
            if 'address' in config_data['domain_id']['ip_address']:
                command += ' {address}'.format(**config_data['domain_id']['ip_address'])
                if 'secondary' in config_data['domain_id']['ip_address']:
                    command += ' secondary'.format(**config_data['domain_id']['ip_address'])
        elif 'null' in config_data['domain_id']:
            command += ' null'
        return command