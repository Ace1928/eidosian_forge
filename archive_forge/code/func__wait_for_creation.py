from copy import deepcopy
from functools import wraps
from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.amazon.aws.plugins.module_utils.tagging import boto3_tag_list_to_ansible_dict
def _wait_for_creation(self):
    if not self._wait:
        return
    params = self._waiter_config
    self._do_creation_wait(**params)