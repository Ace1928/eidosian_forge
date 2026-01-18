from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.amazon.aws.plugins.module_utils.modules import AnsibleAWSModule
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
@AWSRetry.jittered_backoff()
def _paginated_result(paginator_name, **params):
    paginator = client.get_paginator(paginator_name)
    return paginator.paginate(**params).build_full_result()