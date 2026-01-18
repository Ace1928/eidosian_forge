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
def _validate_password_history(self, password, user_ref):
    unique_cnt = CONF.security_compliance.unique_last_password_count
    if unique_cnt > 0:
        for password_ref in user_ref.local_user.passwords[-unique_cnt:]:
            if password_hashing.check_password(password, password_ref.password_hash):
                raise exception.PasswordHistoryValidationError(unique_count=unique_cnt)