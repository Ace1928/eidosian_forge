import uuid
from keystoneauth1.exceptions import http
from keystoneclient.tests.functional import base
from keystoneclient.tests.functional.v3 import client_fixtures as fixtures
def _new_ref(self):
    return {'identity': {'driver': uuid.uuid4().hex}, 'ldap': {'url': uuid.uuid4().hex}}