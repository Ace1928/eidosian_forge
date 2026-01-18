from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import exceptions
from googlecloudsdk.core.util import retry
def RetryOnHttpStatusAttribute(func):
    """Decorator to automatically retry a function for HTTP errors."""

    def retryIf(exc_type, exc_value, unused_traceback, unused_state):
        return exc_type == exceptions.HttpError and exc_value.status_code == status

    def wrapper(*args, **kwargs):
        retryer = retry.Retryer(max_retrials=3, exponential_sleep_multiplier=2, jitter_ms=100)
        return retryer.RetryOnException(func, args, kwargs, should_retry_if=retryIf, sleep_ms=1000)
    return wrapper