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
def _create_password_expires_query(self, session, query, hints):
    for filter_ in hints.filters:
        if 'password_expires_at' == filter_['name']:
            query = query.filter(sqlalchemy.and_(model.LocalUser.id == model.Password.local_user_id, filter_['comparator'](model.Password.expires_at, filter_['value'])))
    hints.filters = [x for x in hints.filters if x['name'] != 'password_expires_at']
    return (query, hints)