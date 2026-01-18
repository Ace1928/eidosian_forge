import re
from ansible.module_utils._text import to_text
from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_code
from ansible_collections.amazon.aws.plugins.module_utils.tagging import boto3_tag_list_to_ansible_dict
from ansible_collections.amazon.aws.plugins.module_utils.transformation import ansible_dict_to_boto3_filter_list
from ansible_collections.amazon.aws.plugins.plugin_utils.inventory import AWSInventoryBase
def _get_tag_hostname(preference, instance):
    tag_hostnames = preference.split('tag:', 1)[1]
    if ',' in tag_hostnames:
        tag_hostnames = tag_hostnames.split(',')
    else:
        tag_hostnames = [tag_hostnames]
    tags = boto3_tag_list_to_ansible_dict(instance.get('Tags', []))
    tag_values = []
    for v in tag_hostnames:
        if '=' in v:
            tag_name, tag_value = v.split('=')
            if tags.get(tag_name) == tag_value:
                tag_values.append(to_text(tag_name) + '_' + to_text(tag_value))
        else:
            tag_value = tags.get(v)
            if tag_value:
                tag_values.append(to_text(tag_value))
    return tag_values