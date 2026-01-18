import http.client
from oslo_log import log
from oslo_utils import encodeutils
import keystone.conf
from keystone.i18n import _
class RegionDeletionError(ForbiddenNotSecurity):
    message_format = _('Unable to delete region %(region_id)s because it or its child regions have associated endpoints.')