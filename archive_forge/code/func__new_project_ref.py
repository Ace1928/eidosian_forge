import fixtures
import uuid
from keystoneauth1 import exceptions as ksa_exceptions
from keystoneclient import exceptions as ksc_exceptions
from keystoneclient.tests.unit.v3 import utils
from keystoneclient.v3 import projects
def _new_project_ref(self, ref=None):
    ref = ref or {}
    ref.setdefault('domain_id', uuid.uuid4().hex)
    ref.setdefault('enabled', True)
    ref.setdefault('name', uuid.uuid4().hex)
    return ref