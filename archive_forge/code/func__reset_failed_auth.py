import datetime
from oslo_db import api as oslo_db_api
import sqlalchemy
from keystone.common import driver_hints
from keystone.common import password_hashing
from keystone.common import resource_options
from keystone.common import sql
import keystone.conf
from keystone import exception
from keystone.i18n import _
from keystone.identity.backends import base
from keystone.identity.backends import resource_options as options
from keystone.identity.backends import sql_model as model
def _reset_failed_auth(self, user_id):
    with sql.session_for_write() as session:
        user_ref = session.get(model.User, user_id)
        user_ref.local_user.failed_auth_count = 0
        user_ref.local_user.failed_auth_at = None