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
def _change_password_required(self, user):
    if not CONF.security_compliance.change_password_upon_first_use:
        return False
    ignore_option = user.get_resource_option(options.IGNORE_CHANGE_PASSWORD_OPT.option_id)
    return not (ignore_option and ignore_option.option_value is True)