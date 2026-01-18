from botocore.vendored import requests
from botocore.vendored.requests.packages import urllib3
def _exception_from_packed_args(exception_cls, args=None, kwargs=None):
    if args is None:
        args = ()
    if kwargs is None:
        kwargs = {}
    return exception_cls(*args, **kwargs)