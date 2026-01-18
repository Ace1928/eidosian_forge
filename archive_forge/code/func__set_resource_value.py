from copy import deepcopy
from functools import wraps
from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.amazon.aws.plugins.module_utils.tagging import boto3_tag_list_to_ansible_dict
def _set_resource_value(self, key, value, description=None, immutable=False):
    if value is None:
        return False
    if value == self._get_resource_value(key):
        return False
    if immutable and self.original_resource:
        if description is None:
            description = key
        self.module.fail_json(msg=f'{description} can not be updated after creation')
    self._resource_updates[key] = value
    self.changed = True
    return True