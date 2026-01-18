from oslo_db import exception as db_exc
from oslo_log import log as logging
import sqlalchemy.orm as sa_orm
from glance.common import exception as exc
from glance.db.sqlalchemy.metadef_api import namespace as namespace_api
from glance.db.sqlalchemy.metadef_api import resource_type as resource_type_api
from glance.db.sqlalchemy.metadef_api import utils as metadef_utils
from glance.db.sqlalchemy import models_metadef as models
def _set_model_dict(resource_type_name, properties_target, prefix, created_at, updated_at):
    """return a model dict set with the passed in key values"""
    model_dict = {'name': resource_type_name, 'properties_target': properties_target, 'prefix': prefix, 'created_at': created_at, 'updated_at': updated_at}
    return model_dict