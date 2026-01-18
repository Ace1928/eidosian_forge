from oslo_db import exception as db_exc
from oslo_db.sqlalchemy.utils import paginate_query
from oslo_log import log as logging
from sqlalchemy import func
import sqlalchemy.orm as sa_orm
from glance.common import exception as exc
from glance.db.sqlalchemy.metadef_api import namespace as namespace_api
import glance.db.sqlalchemy.metadef_api.utils as metadef_utils
from glance.db.sqlalchemy import models_metadef as models
from glance.i18n import _LW
def _get_by_name(context, session, namespace_name, name):
    namespace = namespace_api.get(context, session, namespace_name)
    try:
        query = session.query(models.MetadefTag).filter_by(name=name, namespace_id=namespace['id'])
        metadef_tag = query.one()
    except sa_orm.exc.NoResultFound:
        LOG.debug('The metadata tag with name=%(name)s was not found in namespace=%(namespace_name)s.', {'name': name, 'namespace_name': namespace_name})
        raise exc.MetadefTagNotFound(name=name, namespace_name=namespace_name)
    return metadef_tag