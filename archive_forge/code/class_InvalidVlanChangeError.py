from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.general.plugins.module_utils.dimensiondata import DimensionDataModule, UnknownNetworkError
class InvalidVlanChangeError(Exception):
    """
    Error raised when an illegal change to VLAN state is attempted.
    """
    pass