from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
from ansible.module_utils.common.dict_transformations import _snake_to_camel
from ansible.module_utils.six import string_types

            Default dictionary comparison.
            This function will work well with most of the Azure resources.
            It correctly handles "location" comparison.

            Value handling:
                - if "new" value is None, it will be taken from "old" dictionary if "incremental_update"
                  is enabled.
            List handling:
                - if list contains "name" field it will be sorted by "name" before comparison is done.
                - if module has "incremental_update" set, items missing in the new list will be copied
                  from the old list

            Warnings:
                If field is marked as non-updatable, appropriate warning will be printed out and
                "new" structure will be updated to old value.

            :modifiers: Optional dictionary of modifiers, where key is the path and value is dict of modifiers
            :param new: New version
            :param old: Old version

            Returns True if no difference between structures has been detected.
            Returns False if difference was detected.
        