from functools import wraps
from .cloud import CloudRetry
def _botocore_exception_maybe():
    """
    Allow for boto3 not being installed when using these utils by wrapping
    botocore.exceptions instead of assigning from it directly.
    """
    if HAS_BOTO3:
        return ClientError
    return type(None)