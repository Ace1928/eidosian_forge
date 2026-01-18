from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible.module_utils.common.dict_transformations import snake_dict_to_camel_dict
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_code
from ansible_collections.amazon.aws.plugins.module_utils.ec2 import get_ec2_security_group_ids_from_names
from ansible_collections.amazon.aws.plugins.module_utils.modules import AnsibleAWSModule
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.amazon.aws.plugins.module_utils.tagging import ansible_dict_to_boto3_tag_list
from ansible_collections.amazon.aws.plugins.module_utils.tagging import boto3_tag_list_to_ansible_dict
from ansible_collections.amazon.aws.plugins.module_utils.tagging import compare_aws_tags
from ansible_collections.amazon.aws.plugins.module_utils.transformation import ansible_dict_to_boto3_filter_list
from ansible_collections.amazon.aws.plugins.module_utils.transformation import scrub_none_parameters
from ansible_collections.amazon.aws.plugins.module_utils.waiters import get_waiter
def _format_listener(self, listener, inject_protocol=False):
    """Formats listener into the format needed by the
        ELB API"""
    listener = scrub_none_parameters(listener)
    for protocol in ['protocol', 'instance_protocol']:
        if protocol in listener:
            listener[protocol] = listener[protocol].upper()
    if inject_protocol and 'instance_protocol' not in listener:
        listener['instance_protocol'] = listener['protocol']
    listener.pop('proxy_protocol', None)
    ssl_id = listener.pop('ssl_certificate_id', None)
    formatted_listener = snake_dict_to_camel_dict(listener, True)
    if ssl_id:
        formatted_listener['SSLCertificateId'] = ssl_id
    return formatted_listener