import sqlalchemy
from keystone.common import sql
from keystone.models import revoke_model
from keystone.revoke.backends import base
from oslo_db import api as oslo_db_api
def _prune_expired_events(self):
    oldest = base.revoked_before_cutoff_time()
    with sql.session_for_write() as session:
        dialect = session.bind.dialect.name
        batch_size = self._flush_batch_size(dialect)
        if batch_size > 0:
            query = session.query(RevocationEvent.id)
            query = query.filter(RevocationEvent.revoked_at < oldest)
            query = query.limit(batch_size).subquery()
            delete_query = session.query(RevocationEvent).filter(RevocationEvent.id.in_(query))
            while True:
                rowcount = delete_query.delete(synchronize_session=False)
                if rowcount == 0:
                    break
        else:
            query = session.query(RevocationEvent)
            query = query.filter(RevocationEvent.revoked_at < oldest)
            query.delete(synchronize_session=False)
        session.flush()