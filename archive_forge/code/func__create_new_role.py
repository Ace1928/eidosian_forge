import datetime
import random
import uuid
import freezegun
import http.client
from testtools import matchers
from keystone.common import provider_api
import keystone.conf
from keystone import exception
from keystone.resource.backends import base as resource_base
from keystone.tests import unit
from keystone.tests.unit import test_v3
def _create_new_role(self):
    """Create a role available for use anywhere and return the ID."""
    ref = unit.new_role_ref()
    response = self.post('/roles', body={'role': ref})
    return response.json_body['role']['id']