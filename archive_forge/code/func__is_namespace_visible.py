from oslo_db import exception as db_exc
from oslo_db.sqlalchemy.utils import paginate_query
from oslo_log import log as logging
import sqlalchemy.exc as sa_exc
from sqlalchemy import or_
import sqlalchemy.orm as sa_orm
from glance.common import exception as exc
import glance.db.sqlalchemy.metadef_api as metadef_api
from glance.db.sqlalchemy import models_metadef as models
from glance.i18n import _
def _is_namespace_visible(context, namespace, status=None):
    """Return True if the namespace is visible in this context."""
    if context.is_admin:
        return True
    if namespace['owner'] is None:
        return True
    if 'visibility' in namespace:
        if namespace['visibility'] == 'public':
            return True
    if context.owner is not None:
        if context.owner == namespace['owner']:
            return True
    return False