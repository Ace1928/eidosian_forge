import logging
import time
from containerregistry.transport import nested
import httplib2
import six.moves.http_client
def WithSourceTransportCallable(self, source_transport_callable):
    self.source_transport_callable = source_transport_callable
    return self