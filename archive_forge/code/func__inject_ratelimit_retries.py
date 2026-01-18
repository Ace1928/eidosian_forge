from copy import deepcopy
from functools import wraps
from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.amazon.aws.plugins.module_utils.tagging import boto3_tag_list_to_ansible_dict
def _inject_ratelimit_retries(self, model):
    extra_retries = ['RequestLimitExceeded', 'Unavailable', 'ServiceUnavailable', 'InternalFailure', 'InternalError', 'TooManyRequestsException', 'Throttling']
    acceptors = []
    for error in extra_retries:
        acceptors.append(dict(state='retry', matcher='error', expected=error))
    _model = deepcopy(model)
    for waiter in _model:
        _model[waiter]['acceptors'].extend(acceptors)
    return _model