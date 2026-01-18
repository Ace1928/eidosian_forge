import sqlalchemy
from sqlalchemy.sql import true
from keystone.catalog.backends import base
from keystone.common import driver_hints
from keystone.common import sql
from keystone.common import utils
import keystone.conf
from keystone import exception
from keystone.i18n import _
class ProjectEndpointGroupMembership(sql.ModelBase, sql.ModelDictMixin):
    """Project to Endpoint group relationship table."""
    __tablename__ = 'project_endpoint_group'
    attributes = ['endpoint_group_id', 'project_id']
    endpoint_group_id = sql.Column(sql.String(64), sql.ForeignKey('endpoint_group.id'), nullable=False)
    project_id = sql.Column(sql.String(64), nullable=False)
    __table_args__ = (sql.PrimaryKeyConstraint('endpoint_group_id', 'project_id'),)