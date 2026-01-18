from __future__ import (absolute_import, division, print_function)
import traceback
import time
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.ovirt.ovirt.plugins.module_utils.ovirt import (
def _get_domain_mappings(module):
    domainMappings = list()
    for domainMapping in module.params['domain_mappings']:
        domainMappings.append(otypes.RegistrationDomainMapping(from_=otypes.Domain(name=domainMapping['source_name']) if domainMapping['source_name'] else None, to=otypes.Domain(name=domainMapping['dest_name']) if domainMapping['dest_name'] else None))
    return domainMappings