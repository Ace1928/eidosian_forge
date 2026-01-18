import io
import functools
import logging
import time
import warnings
import smart_open.bytebuffer
import smart_open.concurrency
import smart_open.utils
from smart_open import constants
def _consolidate_params(uri, transport_params):
    """Consolidates the parsed Uri with the additional parameters.

    This is necessary because the user can pass some of the parameters can in
    two different ways:

    1) Via the URI itself
    2) Via the transport parameters

    These are not mutually exclusive, but we have to pick one over the other
    in a sensible way in order to proceed.

    """
    transport_params = dict(transport_params)

    def inject(**kwargs):
        try:
            client_kwargs = transport_params['client_kwargs']
        except KeyError:
            client_kwargs = transport_params['client_kwargs'] = {}
        try:
            init_kwargs = client_kwargs['S3.Client']
        except KeyError:
            init_kwargs = client_kwargs['S3.Client'] = {}
        init_kwargs.update(**kwargs)
    client = transport_params.get('client')
    if client is not None and (uri['access_id'] or uri['access_secret']):
        logger.warning('ignoring credentials parsed from URL because they conflict with transport_params["client"]. Set transport_params["client"] to None to suppress this warning.')
        uri.update(access_id=None, access_secret=None)
    elif uri['access_id'] and uri['access_secret']:
        inject(aws_access_key_id=uri['access_id'], aws_secret_access_key=uri['access_secret'])
        uri.update(access_id=None, access_secret=None)
    if client is not None and uri['host'] != DEFAULT_HOST:
        logger.warning('ignoring endpoint_url parsed from URL because they conflict with transport_params["client"]. Set transport_params["client"] to None to suppress this warning.')
        uri.update(host=None)
    elif uri['host'] != DEFAULT_HOST:
        inject(endpoint_url='https://%(host)s:%(port)d' % uri)
        uri.update(host=None)
    return (uri, transport_params)