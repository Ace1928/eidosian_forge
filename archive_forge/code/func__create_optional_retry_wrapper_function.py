from functools import wraps
from .cloud import CloudRetry
def _create_optional_retry_wrapper_function(self, unwrapped):
    retrying_wrapper = self.retry(unwrapped)

    @wraps(unwrapped)
    def deciding_wrapper(*args, aws_retry=False, **kwargs):
        if aws_retry:
            return retrying_wrapper(*args, **kwargs)
        else:
            return unwrapped(*args, **kwargs)
    return deciding_wrapper