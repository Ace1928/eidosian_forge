import datetime
import uuid
import freezegun
import passlib.hash
from keystone.common import password_hashing
from keystone.common import provider_api
from keystone.common import resource_options
from keystone.common import sql
import keystone.conf
from keystone import exception
from keystone.identity.backends import base
from keystone.identity.backends import resource_options as iro
from keystone.identity.backends import sql_model as model
from keystone.tests.unit import test_backend_sql
def _add_passwords_to_history(self, user, n):
    for _ in range(n):
        user['password'] = uuid.uuid4().hex
        PROVIDERS.identity_api.update_user(user['id'], user)