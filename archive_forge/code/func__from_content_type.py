import http.client
from oslo_serialization import jsonutils
import webtest
from keystone.tests import unit
from keystone.tests.unit import default_fixtures
from keystone.tests.unit import ksfixtures
from keystone.tests.unit.ksfixtures import database
def _from_content_type(self, response, content_type=None):
    """Attempt to decode JSON and XML automatically, if detected."""
    content_type = content_type or self.content_type
    if response.body is not None and response.body.strip():
        header = response.headers.get('Content-Type')
        self.assertIn(content_type, header)
        if content_type == 'json':
            response.result = jsonutils.loads(response.body)
        else:
            response.result = response.body