from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
def account_obj_to_dict_basic(self, datalake_store_obj):
    account_dict = dict(account_id=datalake_store_obj.account_id, creation_time=datalake_store_obj.creation_time, endpoint=datalake_store_obj.endpoint, id=datalake_store_obj.id, last_modified_time=datalake_store_obj.last_modified_time, location=datalake_store_obj.location, name=datalake_store_obj.name, provisioning_state=datalake_store_obj.provisioning_state, state=datalake_store_obj.state, tags=datalake_store_obj.tags, type=datalake_store_obj.type)
    return account_dict