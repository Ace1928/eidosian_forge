from functools import wraps
from .cloud import CloudRetry
class RetryingBotoClientWrapper:
    __never_wait = ('get_paginator', 'can_paginate', 'get_waiter', 'generate_presigned_url')

    def __init__(self, client, retry):
        self.client = client
        self.retry = retry

    def _create_optional_retry_wrapper_function(self, unwrapped):
        retrying_wrapper = self.retry(unwrapped)

        @wraps(unwrapped)
        def deciding_wrapper(*args, aws_retry=False, **kwargs):
            if aws_retry:
                return retrying_wrapper(*args, **kwargs)
            else:
                return unwrapped(*args, **kwargs)
        return deciding_wrapper

    def __getattr__(self, name):
        unwrapped = getattr(self.client, name)
        if name in self.__never_wait:
            return unwrapped
        elif callable(unwrapped):
            wrapped = self._create_optional_retry_wrapper_function(unwrapped)
            setattr(self, name, wrapped)
            return wrapped
        else:
            return unwrapped