import logging
import time
from containerregistry.transport import nested
import httplib2
import six.moves.http_client
def WithBackoffFactor(self, backoff_factor):
    self.kwargs['backoff_factor'] = backoff_factor
    return self