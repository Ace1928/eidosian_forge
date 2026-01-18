from oslo_log import log
from oslo_serialization import jsonutils
from sqlalchemy import orm
from keystone.common import sql
import keystone.conf
from keystone import exception
from keystone.federation.backends import base
from keystone.i18n import _
def _get_protocol(self, session, idp_id, protocol_id):
    q = session.query(FederationProtocolModel)
    q = q.filter_by(id=protocol_id, idp_id=idp_id)
    try:
        return q.one()
    except sql.NotFound:
        kwargs = {'protocol_id': protocol_id, 'idp_id': idp_id}
        raise exception.FederatedProtocolNotFound(**kwargs)