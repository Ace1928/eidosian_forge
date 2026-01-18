from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase

        Transform cluster profile object to dict
        :param: profile: ManagedClusterUpgradeProfile with AKS upgrade profile info
        :return: dict with upgrade profiles
        