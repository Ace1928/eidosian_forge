import sqlalchemy
from sqlalchemy.sql import true
from keystone.catalog.backends import base
from keystone.common import driver_hints
from keystone.common import sql
from keystone.common import utils
import keystone.conf
from keystone import exception
from keystone.i18n import _
def _has_endpoints(self, session, region, root_region):
    if region.endpoints is not None and len(region.endpoints) > 0:
        return True
    q = session.query(Region)
    q = q.filter_by(parent_region_id=region.id)
    for child in q.all():
        if child.id == root_region.id:
            return False
        if self._has_endpoints(session, child, root_region):
            return True
    return False