import re
from ansible.module_utils._text import to_text
from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_code
from ansible_collections.amazon.aws.plugins.module_utils.tagging import boto3_tag_list_to_ansible_dict
from ansible_collections.amazon.aws.plugins.module_utils.transformation import ansible_dict_to_boto3_filter_list
from ansible_collections.amazon.aws.plugins.plugin_utils.inventory import AWSInventoryBase
def _compile_values(obj, attr):
    """
    :param obj: A list or dict of instance attributes
    :param attr: A key
    :return The value(s) found via the attr
    """
    if obj is None:
        return
    temp_obj = []
    if isinstance(obj, list) or isinstance(obj, tuple):
        for each in obj:
            value = _compile_values(each, attr)
            if value:
                temp_obj.append(value)
    else:
        temp_obj = obj.get(attr)
    has_indexes = any([isinstance(temp_obj, list), isinstance(temp_obj, tuple)])
    if has_indexes and len(temp_obj) == 1:
        return temp_obj[0]
    return temp_obj