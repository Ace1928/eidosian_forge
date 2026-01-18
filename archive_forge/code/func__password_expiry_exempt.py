import datetime
import sqlalchemy
from sqlalchemy.ext.hybrid import hybrid_property
from sqlalchemy import orm
from sqlalchemy.orm import collections
from keystone.common import password_hashing
from keystone.common import resource_options
from keystone.common import sql
import keystone.conf
from keystone.identity.backends import resource_options as iro
def _password_expiry_exempt(self):
    return getattr(self.get_resource_option(iro.IGNORE_PASSWORD_EXPIRY_OPT.option_id), 'option_value', False)