from oslo_log import log
from oslo_serialization import jsonutils
from sqlalchemy import orm
from keystone.common import sql
import keystone.conf
from keystone import exception
from keystone.federation.backends import base
from keystone.i18n import _
def _delete_assigned_protocols(self, session, idp_id):
    query = session.query(FederationProtocolModel)
    query = query.filter_by(idp_id=idp_id)
    query.delete()