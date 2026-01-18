from oslo_db import exception as db_exception
from keystone.assignment.role_backends import base
from keystone.assignment.role_backends import sql_model
from keystone.common import driver_hints
from keystone.common import resource_options
from keystone.common import sql
from keystone import exception
def _get_implied_role(self, session, prior_role_id, implied_role_id):
    query = session.query(sql_model.ImpliedRoleTable).filter(sql_model.ImpliedRoleTable.prior_role_id == prior_role_id).filter(sql_model.ImpliedRoleTable.implied_role_id == implied_role_id)
    try:
        ref = query.one()
    except sql.NotFound:
        raise exception.ImpliedRoleNotFound(prior_role_id=prior_role_id, implied_role_id=implied_role_id)
    return ref