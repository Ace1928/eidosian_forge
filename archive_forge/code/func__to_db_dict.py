from oslo_db import exception as db_exc
from oslo_log import log as logging
import sqlalchemy.orm as sa_orm
from glance.common import exception as exc
from glance.db.sqlalchemy.metadef_api import namespace as namespace_api
from glance.db.sqlalchemy.metadef_api import resource_type as resource_type_api
from glance.db.sqlalchemy.metadef_api import utils as metadef_utils
from glance.db.sqlalchemy import models_metadef as models
def _to_db_dict(namespace_id, resource_type_id, model_dict):
    """transform a model dict to a metadef_namespace_resource_type dict"""
    db_dict = {'namespace_id': namespace_id, 'resource_type_id': resource_type_id, 'properties_target': model_dict['properties_target'], 'prefix': model_dict['prefix']}
    return db_dict