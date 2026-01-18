from __future__ import absolute_import, division, print_function
import copy
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AZURE_SUCCESS_STATE, AzureRMModuleBase
from ansible.module_utils._text import to_native

        If there are blob containers, then there are likely VMs depending on this account and it should
        not be deleted.
        