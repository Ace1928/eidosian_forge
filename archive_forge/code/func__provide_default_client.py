import functools
from qcs_api_client.client import build_sync_client
def _provide_default_client(function):
    """A decorator that will initialize an `httpx.Client` and pass
    it to the wrapped function as a kwarg if not already present. This
    eases provision of a default `httpx.Client` with Rigetti
    QCS configuration and authentication. If the decorator initializes a
    default client, it will invoke the wrapped function from within the
    `httpx.Client` context.

    Args:
        function: The decorated function.

    Returns:
        The `function` wrapped with a default `client`.
    """

    @functools.wraps(function)
    def wrapper(*args, **kwargs):
        if 'client' in kwargs:
            return function(*args, **kwargs)
        with build_sync_client() as client:
            kwargs['client'] = client
            return function(*args, **kwargs)
    return wrapper