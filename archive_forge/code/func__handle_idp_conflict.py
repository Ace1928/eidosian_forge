from oslo_log import log
from oslo_serialization import jsonutils
from sqlalchemy import orm
from keystone.common import sql
import keystone.conf
from keystone import exception
from keystone.federation.backends import base
from keystone.i18n import _
def _handle_idp_conflict(self, e):
    conflict_type = 'identity_provider'
    details = str(e)
    LOG.debug(self._CONFLICT_LOG_MSG, {'conflict_type': conflict_type, 'details': details})
    if 'remote_id' in details:
        msg = _('Duplicate remote ID: %s')
    else:
        msg = _('Duplicate entry: %s')
    msg = msg % e.value
    raise exception.Conflict(type=conflict_type, details=msg)