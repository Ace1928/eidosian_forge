import sqlalchemy
from sqlalchemy.sql import true
from keystone.catalog.backends import base
from keystone.common import driver_hints
from keystone.common import sql
from keystone.common import utils
import keystone.conf
from keystone import exception
from keystone.i18n import _
def _delete_endpoint_group_association_by_endpoint_group(self, session, endpoint_group_id):
    query = session.query(ProjectEndpointGroupMembership)
    query = query.filter_by(endpoint_group_id=endpoint_group_id)
    query.delete()