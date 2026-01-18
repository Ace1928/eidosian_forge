from oslo_log import log
from sqlalchemy import orm
from sqlalchemy.sql import expression
from keystone.common import driver_hints
from keystone.common import resource_options
from keystone.common import sql
from keystone import exception
from keystone.resource.backends import base
from keystone.resource.backends import sql_model
def _is_hidden_ref(self, ref):
    return ref.id == base.NULL_DOMAIN_ID