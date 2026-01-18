import logging
import time
from containerregistry.transport import nested
import httplib2
import six.moves.http_client
def WithShouldRetryFunction(self, should_retry_fn):
    self.kwargs['should_retry_fn'] = should_retry_fn
    return self