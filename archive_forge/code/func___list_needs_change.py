from time import sleep
from time import time
from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.amazon.aws.plugins.module_utils.modules import AnsibleAWSModule
def __list_needs_change(current, desired):
    if len(current) != len(desired):
        return True
    c_sorted = sorted(current)
    d_sorted = sorted(desired)
    for index, value in enumerate(current):
        if value != desired[index]:
            return True
    return False