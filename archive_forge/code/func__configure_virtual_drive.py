from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.cisco.ucs.plugins.module_utils.ucs import UCSModule, ucs_argument_spec
def _configure_virtual_drive(module, mo):
    from ucsmsdk.mometa.lstorage.LstorageVirtualDriveDef import LstorageVirtualDriveDef
    LstorageVirtualDriveDef(parent_mo_or_dn=mo, **module.params['virtual_drive'])