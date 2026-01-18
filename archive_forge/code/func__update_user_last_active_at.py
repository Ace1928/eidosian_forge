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
def _update_user_last_active_at(self, user_id, last_active_at):
    with sql.session_for_write() as session:
        user_ref = session.get(model.User, user_id)
        user_ref.last_active_at = last_active_at
        return user_ref