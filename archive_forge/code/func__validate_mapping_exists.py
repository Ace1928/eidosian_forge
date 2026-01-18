import uuid
from oslo_log import log
from keystone.common import cache
from keystone.common import driver_hints
from keystone.common import manager
from keystone.common import provider_api
import keystone.conf
from keystone import exception
from keystone.federation import utils
from keystone.i18n import _
from keystone import notifications
def _validate_mapping_exists(self, mapping_id):
    try:
        self.driver.get_mapping(mapping_id)
    except exception.MappingNotFound:
        msg = _('Invalid mapping id: %s')
        raise exception.ValidationError(message=msg % mapping_id)