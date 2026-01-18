import sqlalchemy
from keystone.common import sql
from keystone.models import revoke_model
from keystone.revoke.backends import base
from oslo_db import api as oslo_db_api
def _list_last_fetch_events(self, last_fetch=None):
    with sql.session_for_read() as session:
        query = session.query(RevocationEvent).order_by(RevocationEvent.revoked_at)
        if last_fetch:
            query = query.filter(RevocationEvent.revoked_at > last_fetch)
        events = [revoke_model.RevokeEvent(**e.to_dict()) for e in query]
        return events