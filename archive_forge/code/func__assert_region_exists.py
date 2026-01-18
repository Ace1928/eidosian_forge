from keystone.common import cache
from keystone.common import driver_hints
from keystone.common import manager
from keystone.common import provider_api
import keystone.conf
from keystone import exception
from keystone.i18n import _
from keystone import notifications
def _assert_region_exists(self, region_id):
    try:
        if region_id is not None:
            self.get_region(region_id)
    except exception.RegionNotFound:
        raise exception.ValidationError(attribute='endpoint region_id', target='region table')