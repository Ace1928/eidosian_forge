import collections
import grpc
from google.api_core import exceptions
from google.api_core import retry
from google.api_core import timeout
def _timeout_from_retry_config(retry_params):
    """Creates a ExponentialTimeout object given a gapic retry configuration.

    DEPRECATED: instantiate retry and timeout classes directly instead.

    Args:
        retry_params (dict): The retry parameter values, for example::

            {
                "initial_retry_delay_millis": 1000,
                "retry_delay_multiplier": 2.5,
                "max_retry_delay_millis": 120000,
                "initial_rpc_timeout_millis": 120000,
                "rpc_timeout_multiplier": 1.0,
                "max_rpc_timeout_millis": 120000,
                "total_timeout_millis": 600000
            }

    Returns:
        google.api_core.retry.ExponentialTimeout: The default time object for
            the method.
    """
    return timeout.ExponentialTimeout(initial=retry_params['initial_rpc_timeout_millis'] / _MILLIS_PER_SECOND, maximum=retry_params['max_rpc_timeout_millis'] / _MILLIS_PER_SECOND, multiplier=retry_params['rpc_timeout_multiplier'], deadline=retry_params['total_timeout_millis'] / _MILLIS_PER_SECOND)