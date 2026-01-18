import copy
import sqlalchemy
from sqlalchemy.ext.hybrid import hybrid_property
from oslo_db import exception as db_exception
from keystone.common import driver_hints
from keystone.common import sql
from keystone import exception
from keystone.i18n import _
from keystone.limit.backends import base
def _get_registered_limit(self, session, registered_limit_id):
    query = session.query(RegisteredLimitModel).filter_by(id=registered_limit_id)
    ref = query.first()
    if ref is None:
        raise exception.RegisteredLimitNotFound(id=registered_limit_id)
    return ref