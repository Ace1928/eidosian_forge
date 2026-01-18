from time import sleep
from time import time
from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.amazon.aws.plugins.module_utils.modules import AnsibleAWSModule
def _needs_change(current, desired):
    needs_change = False
    for key in desired:
        current_value = current[key]
        desired_value = desired[key]
        if isinstance(current_value, (int, str, bool)):
            if current_value != desired_value:
                needs_change = True
                break
        elif isinstance(current_value, list):
            if __list_needs_change(current_value, desired_value):
                needs_change = True
                break
        elif isinstance(current_value, dict):
            if __dict_needs_change(current_value, desired_value):
                needs_change = True
                break
        else:
            needs_change = True
            break
    return needs_change