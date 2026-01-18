import time
from ansible_collections.amazon.aws.plugins.module_utils.modules import AnsibleAWSModule
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.amazon.aws.plugins.module_utils.route53 import get_tags
from ansible_collections.amazon.aws.plugins.module_utils.route53 import manage_tags
@AWSRetry.jittered_backoff()
def _list_zones():
    paginator = client.get_paginator('list_hosted_zones')
    return paginator.paginate().build_full_result()