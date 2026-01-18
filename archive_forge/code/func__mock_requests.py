from unittest import mock
import requests
import glance_store
from glance_store._drivers import http
from glance_store import exceptions
from glance_store import location
from glance_store.tests import base
from glance_store.tests.unit import test_store_capabilities
from glance_store.tests import utils
def _mock_requests(self):
    """Mock requests session object.

        Should be called when we need to mock request/response objects.
        """
    request = mock.patch('requests.Session.request')
    self.request = request.start()
    self.addCleanup(request.stop)