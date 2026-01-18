import copy
import sqlalchemy
from sqlalchemy.ext.hybrid import hybrid_property
from oslo_db import exception as db_exception
from keystone.common import driver_hints
from keystone.common import sql
from keystone import exception
from keystone.i18n import _
from keystone.limit.backends import base
def _check_unified_limit_unique(self, unified_limit, is_registered_limit=True):
    hints = driver_hints.Hints()
    if is_registered_limit:
        hints.add_filter('service_id', unified_limit['service_id'])
        hints.add_filter('resource_name', unified_limit['resource_name'])
        hints.add_filter('region_id', unified_limit.get('region_id'))
        with sql.session_for_read() as session:
            query = session.query(RegisteredLimitModel)
            unified_limits = sql.filter_limit_query(RegisteredLimitModel, query, hints).all()
    else:
        hints.add_filter('registered_limit_id', unified_limit['registered_limit_id'])
        is_project_limit = True if unified_limit.get('project_id') else False
        if is_project_limit:
            hints.add_filter('project_id', unified_limit['project_id'])
        else:
            hints.add_filter('domain_id', unified_limit['domain_id'])
        with sql.session_for_read() as session:
            query = session.query(LimitModel)
            unified_limits = sql.filter_limit_query(LimitModel, query, hints).all()
    if unified_limits:
        msg = _('Duplicate entry')
        limit_type = 'registered_limit' if is_registered_limit else 'limit'
        raise exception.Conflict(type=limit_type, details=msg)