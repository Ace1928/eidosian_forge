from time import sleep
from time import time
from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.amazon.aws.plugins.module_utils.modules import AnsibleAWSModule
def __dict_needs_change(current, desired):
    for key in desired:
        if key in current:
            if desired[key] != current[key]:
                return True
    return False