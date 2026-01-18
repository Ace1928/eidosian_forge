from oslo_db import exception as db_exc
from oslo_log import log as logging
import sqlalchemy.orm as sa_orm
from glance.common import exception as exc
from glance.db.sqlalchemy.metadef_api import namespace as namespace_api
from glance.db.sqlalchemy.metadef_api import resource_type as resource_type_api
from glance.db.sqlalchemy.metadef_api import utils as metadef_utils
from glance.db.sqlalchemy import models_metadef as models
def _to_model_dict(resource_type_name, ns_res_type_dict):
    """transform a metadef_namespace_resource_type dict to a model dict"""
    model_dict = {'name': resource_type_name, 'properties_target': ns_res_type_dict['properties_target'], 'prefix': ns_res_type_dict['prefix'], 'created_at': ns_res_type_dict['created_at'], 'updated_at': ns_res_type_dict['updated_at']}
    return model_dict