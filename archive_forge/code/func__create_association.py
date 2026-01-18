from oslo_db import exception as db_exc
from oslo_log import log as logging
import sqlalchemy.orm as sa_orm
from glance.common import exception as exc
from glance.db.sqlalchemy.metadef_api import namespace as namespace_api
from glance.db.sqlalchemy.metadef_api import resource_type as resource_type_api
from glance.db.sqlalchemy.metadef_api import utils as metadef_utils
from glance.db.sqlalchemy import models_metadef as models
def _create_association(context, session, namespace_name, resource_type_name, values):
    """Create an association, raise if it already exists."""
    namespace_resource_type_rec = models.MetadefNamespaceResourceType()
    metadef_utils.drop_protected_attrs(models.MetadefNamespaceResourceType, values)
    namespace_resource_type_rec.update(values.copy())
    try:
        namespace_resource_type_rec.save(session=session)
    except db_exc.DBDuplicateEntry:
        LOG.debug('The metadata definition resource-type association of resource_type=%(resource_type_name)s to namespace=%(namespace_name)s, already exists.', {'resource_type_name': resource_type_name, 'namespace_name': namespace_name})
        raise exc.MetadefDuplicateResourceTypeAssociation(resource_type_name=resource_type_name, namespace_name=namespace_name)
    return namespace_resource_type_rec.to_dict()