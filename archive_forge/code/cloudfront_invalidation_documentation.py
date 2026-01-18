import datetime
from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible.module_utils.common.dict_transformations import snake_dict_to_camel_dict
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_message
from ansible_collections.amazon.aws.plugins.module_utils.cloudfront_facts import CloudFrontFactsServiceManager
from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule

    Manages CloudFront validations for invalidation batches
    