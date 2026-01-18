import base64
import hashlib
import re
import traceback
from collections import Counter
from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_code
from ansible_collections.amazon.aws.plugins.module_utils.iam import get_aws_account_info
from ansible_collections.amazon.aws.plugins.module_utils.modules import AnsibleAWSModule
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.amazon.aws.plugins.module_utils.tagging import compare_aws_tags
def _code_args(module, current_config):
    s3_bucket = module.params.get('s3_bucket')
    s3_key = module.params.get('s3_key')
    s3_object_version = module.params.get('s3_object_version')
    zip_file = module.params.get('zip_file')
    architectures = module.params.get('architecture')
    image_uri = module.params.get('image_uri')
    code_kwargs = {}
    if architectures and current_config.get('Architectures', None) != [architectures]:
        module.warn('Arch Change')
        code_kwargs.update({'Architectures': [architectures]})
    try:
        code_kwargs.update(_zip_args(zip_file, current_config, bool(code_kwargs)))
    except IOError as e:
        module.fail_json(msg=str(e), exception=traceback.format_exc())
    code_kwargs.update(_s3_args(s3_bucket, s3_key, s3_object_version))
    code_kwargs.update(_image_args(image_uri))
    if not code_kwargs:
        return {}
    if not architectures and current_config.get('Architectures', None):
        code_kwargs.update({'Architectures': current_config.get('Architectures', None)})
    return code_kwargs