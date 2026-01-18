from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
from ansible.module_utils.common.dict_transformations import _snake_to_camel

        Stops an existing Azure CDN endpoint that is on a running state.

        :return: deserialized Azure CDN endpoint state dictionary
        