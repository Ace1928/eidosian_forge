import sqlalchemy
from sqlalchemy.sql import true
from keystone.catalog.backends import base
from keystone.common import driver_hints
from keystone.common import sql
from keystone.common import utils
import keystone.conf
from keystone import exception
from keystone.i18n import _
def _get_endpoint_group(self, session, endpoint_group_id):
    endpoint_group_ref = session.get(EndpointGroup, endpoint_group_id)
    if endpoint_group_ref is None:
        raise exception.EndpointGroupNotFound(endpoint_group_id=endpoint_group_id)
    return endpoint_group_ref