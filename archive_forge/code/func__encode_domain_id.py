from oslo_log import log
from sqlalchemy import orm
from sqlalchemy.sql import expression
from keystone.common import driver_hints
from keystone.common import resource_options
from keystone.common import sql
from keystone import exception
from keystone.resource.backends import base
from keystone.resource.backends import sql_model
def _encode_domain_id(self, ref):
    if 'domain_id' in ref and ref['domain_id'] is None:
        new_ref = ref.copy()
        new_ref['domain_id'] = base.NULL_DOMAIN_ID
        return new_ref
    else:
        return ref