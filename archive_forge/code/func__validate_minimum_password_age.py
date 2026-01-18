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
def _validate_minimum_password_age(self, user_ref):
    min_age_days = CONF.security_compliance.minimum_password_age
    min_age = user_ref.password_created_at + datetime.timedelta(days=min_age_days)
    if datetime.datetime.utcnow() < min_age:
        days_left = (min_age - datetime.datetime.utcnow()).days
        raise exception.PasswordAgeValidationError(min_age_days=min_age_days, days_left=days_left)