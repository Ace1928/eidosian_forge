from oslo_log import log
from sqlalchemy import orm
from sqlalchemy.sql import expression
from keystone.common import driver_hints
from keystone.common import resource_options
from keystone.common import sql
from keystone import exception
from keystone.resource.backends import base
from keystone.resource.backends import sql_model
def _filter_not_tags(self, session, filtered_ids, blacklist_ids):
    subq = session.query(sql_model.Project)
    valid_ids = [q['id'] for q in subq if q['id'] not in blacklist_ids]
    if filtered_ids:
        valid_ids = list(set(valid_ids) & set(filtered_ids))
    return valid_ids