from copy import deepcopy
from functools import wraps
from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.amazon.aws.plugins.module_utils.tagging import boto3_tag_list_to_ansible_dict
def _flush_create(self):
    changed = True
    if not self.module.check_mode:
        changed = self._do_create_resource()
        self._wait_for_creation()
        self._do_creation_wait()
        self.updated_resource = self.get_resource()
    else:
        self.updated_resource = self._normalize_resource(self._generate_updated_resource())
    self._resource_updates = dict()
    self.changed = changed
    return True