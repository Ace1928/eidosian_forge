import datetime
from unittest import mock
import uuid
from keystone.common import provider_api
from keystone.common import sql
import keystone.conf
from keystone import exception
from keystone.identity.backends import sql_model as model
from keystone.identity.shadow_backends import sql as shadow_sql
from keystone.tests import unit
def _add_nonlocal_user(self, nonlocal_user):
    with sql.session_for_write() as session:
        nonlocal_user_ref = model.NonLocalUser.from_dict(nonlocal_user)
        session.add(nonlocal_user_ref)