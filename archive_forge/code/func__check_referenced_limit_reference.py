import copy
import sqlalchemy
from sqlalchemy.ext.hybrid import hybrid_property
from oslo_db import exception as db_exception
from keystone.common import driver_hints
from keystone.common import sql
from keystone import exception
from keystone.i18n import _
from keystone.limit.backends import base
def _check_referenced_limit_reference(self, registered_limit):
    with sql.session_for_read() as session:
        limits = session.query(LimitModel).filter_by(registered_limit_id=registered_limit['id'])
    if limits.all():
        raise exception.RegisteredLimitError(id=registered_limit.id)