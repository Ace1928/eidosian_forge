import logging
import os
import re
import traceback
from ansible.module_utils._text import to_native
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.basic import env_fallback
from ansible.module_utils.basic import missing_required_lib
from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from .botocore import boto3_at_least
from .botocore import boto3_conn
from .botocore import botocore_at_least
from .botocore import check_sdk_version_supported
from .botocore import gather_sdk_versions
from .botocore import get_aws_connection_info
from .botocore import get_aws_region
from .exceptions import AnsibleBotocoreError
from .retries import RetryingBotoClientWrapper
def _aws_common_argument_spec():
    """
    This does not include 'region' as some AWS APIs don't require a
    region.  However, it's not recommended to do this as it means module_defaults
    can't include the region parameter.
    """
    return dict(access_key=dict(aliases=['aws_access_key_id', 'aws_access_key', 'ec2_access_key'], deprecated_aliases=[dict(name='ec2_access_key', date='2024-12-01', collection_name='amazon.aws')], fallback=(env_fallback, ['AWS_ACCESS_KEY_ID', 'AWS_ACCESS_KEY', 'EC2_ACCESS_KEY']), no_log=False), secret_key=dict(aliases=['aws_secret_access_key', 'aws_secret_key', 'ec2_secret_key'], deprecated_aliases=[dict(name='ec2_secret_key', date='2024-12-01', collection_name='amazon.aws')], fallback=(env_fallback, ['AWS_SECRET_ACCESS_KEY', 'AWS_SECRET_KEY', 'EC2_SECRET_KEY']), no_log=True), session_token=dict(aliases=['aws_session_token', 'security_token', 'access_token', 'aws_security_token'], deprecated_aliases=[dict(name='access_token', date='2024-12-01', collection_name='amazon.aws'), dict(name='security_token', date='2024-12-01', collection_name='amazon.aws'), dict(name='aws_security_token', date='2024-12-01', collection_name='amazon.aws')], fallback=(env_fallback, ['AWS_SESSION_TOKEN', 'AWS_SECURITY_TOKEN', 'EC2_SECURITY_TOKEN']), no_log=True), profile=dict(aliases=['aws_profile'], fallback=(env_fallback, ['AWS_PROFILE', 'AWS_DEFAULT_PROFILE'])), endpoint_url=dict(aliases=['aws_endpoint_url', 'ec2_url', 's3_url'], deprecated_aliases=[dict(name='ec2_url', date='2024-12-01', collection_name='amazon.aws'), dict(name='s3_url', date='2024-12-01', collection_name='amazon.aws')], fallback=(env_fallback, ['AWS_URL', 'EC2_URL', 'S3_URL'])), validate_certs=dict(type='bool', default=True), aws_ca_bundle=dict(type='path', fallback=(env_fallback, ['AWS_CA_BUNDLE'])), aws_config=dict(type='dict'), debug_botocore_endpoint_logs=dict(type='bool', default=False, fallback=(env_fallback, ['ANSIBLE_DEBUG_BOTOCORE_LOGS'])))