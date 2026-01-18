import collections
import contextlib
import logging
import socket
import time
import httplib2
import six
from six.moves import http_client
from six.moves.urllib import parse
from apitools.base.py import exceptions
from apitools.base.py import util
def HandleExceptionsAndRebuildHttpConnections(retry_args):
    """Exception handler for http failures.

    This catches known failures and rebuilds the underlying HTTP connections.

    Args:
      retry_args: An ExceptionRetryArgs tuple.
    """
    retry_after = None
    if isinstance(retry_args.exc, (http_client.BadStatusLine, http_client.IncompleteRead, http_client.ResponseNotReady)):
        logging.debug('Caught HTTP error %s, retrying: %s', type(retry_args.exc).__name__, retry_args.exc)
    elif isinstance(retry_args.exc, socket.error):
        logging.debug('Caught socket error, retrying: %s', retry_args.exc)
    elif isinstance(retry_args.exc, socket.gaierror):
        logging.debug('Caught socket address error, retrying: %s', retry_args.exc)
    elif isinstance(retry_args.exc, socket.timeout):
        logging.debug('Caught socket timeout error, retrying: %s', retry_args.exc)
    elif isinstance(retry_args.exc, httplib2.ServerNotFoundError):
        logging.debug('Caught server not found error, retrying: %s', retry_args.exc)
    elif isinstance(retry_args.exc, ValueError):
        logging.debug('Response content was invalid (%s), retrying', retry_args.exc)
    elif isinstance(retry_args.exc, TokenRefreshError) and hasattr(retry_args.exc, 'status') and (retry_args.exc.status == TOO_MANY_REQUESTS or retry_args.exc.status >= 500):
        logging.debug('Caught transient credential refresh error (%s), retrying', retry_args.exc)
    elif isinstance(retry_args.exc, exceptions.RequestError):
        logging.debug('Request returned no response, retrying')
    elif isinstance(retry_args.exc, exceptions.BadStatusCodeError):
        logging.debug('Response returned status %s, retrying', retry_args.exc.status_code)
    elif isinstance(retry_args.exc, exceptions.RetryAfterError):
        logging.debug('Response returned a retry-after header, retrying')
        retry_after = retry_args.exc.retry_after
    else:
        raise retry_args.exc
    RebuildHttpConnections(retry_args.http)
    logging.debug('Retrying request to url %s after exception %s', retry_args.http_request.url, retry_args.exc)
    time.sleep(retry_after or util.CalculateWaitForRetry(retry_args.num_retries, max_wait=retry_args.max_retry_wait))