import copy
import time
from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_code
from ansible_collections.amazon.aws.plugins.module_utils.ec2 import get_ec2_security_group_ids_from_names
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
def _get_glue_connection(connection, module):
    """
    Get an AWS Glue connection based on name. If not found, return None.

    :param connection: AWS boto3 glue connection
    :param module: Ansible module
    :return: boto3 Glue connection dict or None if not found
    """
    connection_name = module.params.get('name')
    connection_catalog_id = module.params.get('catalog_id')
    params = {'Name': connection_name}
    if connection_catalog_id is not None:
        params['CatalogId'] = connection_catalog_id
    try:
        return connection.get_connection(aws_retry=True, **params)['Connection']
    except is_boto3_error_code('EntityNotFoundException'):
        return None