from functools import partial
from ansible.module_utils.common.dict_transformations import snake_dict_to_camel_dict
from .retries import AWSRetry
from .tagging import boto3_tag_list_to_ansible_dict
@AWSRetry.jittered_backoff()
def _cloudfront_paginate_build_full_result(client, client_method, **kwargs):
    paginator = client.get_paginator(client_method)
    return paginator.paginate(**kwargs).build_full_result()