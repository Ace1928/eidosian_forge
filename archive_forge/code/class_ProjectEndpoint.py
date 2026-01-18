import sqlalchemy
from sqlalchemy.sql import true
from keystone.catalog.backends import base
from keystone.common import driver_hints
from keystone.common import sql
from keystone.common import utils
import keystone.conf
from keystone import exception
from keystone.i18n import _
class ProjectEndpoint(sql.ModelBase, sql.ModelDictMixin):
    """project-endpoint relationship table."""
    __tablename__ = 'project_endpoint'
    attributes = ['endpoint_id', 'project_id']
    endpoint_id = sql.Column(sql.String(64), primary_key=True, nullable=False)
    project_id = sql.Column(sql.String(64), primary_key=True, nullable=False)