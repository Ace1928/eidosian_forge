import re
from ansible.module_utils._text import to_text
from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_code
from ansible_collections.amazon.aws.plugins.module_utils.tagging import boto3_tag_list_to_ansible_dict
from ansible_collections.amazon.aws.plugins.module_utils.transformation import ansible_dict_to_boto3_filter_list
from ansible_collections.amazon.aws.plugins.plugin_utils.inventory import AWSInventoryBase
def _get_multiple_ssm_inventories(self, connection, instance_ids):
    result = {}
    while len(instance_ids) > 40:
        filters = [{'Key': 'AWS:InstanceInformation.InstanceId', 'Values': instance_ids[:40]}]
        result.update(_get_ssm_information(connection, filters))
        instance_ids = instance_ids[40:]
    if instance_ids:
        filters = [{'Key': 'AWS:InstanceInformation.InstanceId', 'Values': instance_ids}]
        result.update(_get_ssm_information(connection, filters))
    return result