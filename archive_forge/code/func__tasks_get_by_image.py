import datetime
import itertools
import threading
from oslo_config import cfg
from oslo_db import exception as db_exception
from oslo_db.sqlalchemy import session as oslo_db_session
from oslo_log import log as logging
from oslo_utils import excutils
import osprofiler.sqlalchemy
from retrying import retry
import sqlalchemy
from sqlalchemy.ext.compiler import compiles
from sqlalchemy import MetaData, Table
import sqlalchemy.orm as sa_orm
from sqlalchemy import sql
import sqlalchemy.sql as sa_sql
from glance.common import exception
from glance.common import timeutils
from glance.common import utils
from glance.db.sqlalchemy.metadef_api import (resource_type
from glance.db.sqlalchemy.metadef_api import (resource_type_association
from glance.db.sqlalchemy.metadef_api import namespace as metadef_namespace_api
from glance.db.sqlalchemy.metadef_api import object as metadef_object_api
from glance.db.sqlalchemy.metadef_api import property as metadef_property_api
from glance.db.sqlalchemy.metadef_api import tag as metadef_tag_api
from glance.db.sqlalchemy import models
from glance.db import utils as db_utils
from glance.i18n import _, _LW, _LI, _LE
def _tasks_get_by_image(context, session, image_id):
    tasks = []
    _task_soft_delete(context, session)
    query = session.query(models.Task).options(sa_orm.joinedload(models.Task.info)).filter_by(image_id=image_id)
    expires_at = models.Task.expires_at
    query = query.filter(sa_sql.or_(expires_at == None, expires_at >= timeutils.utcnow()))
    updated_at = models.Task.updated_at
    query.filter(updated_at <= timeutils.utcnow() + datetime.timedelta(hours=CONF.task.task_time_to_live))
    if not context.can_see_deleted:
        query = query.filter_by(deleted=False)
    try:
        task_refs = query.all()
    except sa_orm.exc.NoResultFound:
        LOG.debug('No task found for image with ID %s', image_id)
        return tasks
    for task_ref in task_refs:
        if not _is_task_visible(context, task_ref):
            msg = 'Task %s is not visible, excluding' % task_ref.id
            LOG.debug(msg)
            continue
        tasks.append(_task_format(task_ref, task_ref.info))
    return tasks