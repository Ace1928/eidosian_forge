from boto.compat import http_client
from tests.compat import mock, unittest
def _mexe_spy(self, request, *args, **kwargs):
    self.actual_request = request
    return self.original_mexe(request, *args, **kwargs)