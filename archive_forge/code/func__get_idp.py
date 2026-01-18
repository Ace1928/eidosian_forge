from oslo_log import log
from oslo_serialization import jsonutils
from sqlalchemy import orm
from keystone.common import sql
import keystone.conf
from keystone import exception
from keystone.federation.backends import base
from keystone.i18n import _
def _get_idp(self, session, idp_id):
    idp_ref = session.get(IdentityProviderModel, idp_id)
    if not idp_ref:
        raise exception.IdentityProviderNotFound(idp_id=idp_id)
    return idp_ref