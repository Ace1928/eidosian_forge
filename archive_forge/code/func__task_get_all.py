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
def _task_get_all(context, session, filters=None, marker=None, limit=None, sort_key='created_at', sort_dir='desc', admin_as_user=False):
    filters = filters or {}
    query = session.query(models.Task)
    if not (context.is_admin or admin_as_user) and context.owner is not None:
        query = query.filter(models.Task.owner == context.owner)
    _task_soft_delete(context, session)
    showing_deleted = False
    if 'deleted' in filters:
        deleted_filter = filters.pop('deleted')
        query = query.filter_by(deleted=deleted_filter)
        showing_deleted = deleted_filter
    for k, v in filters.items():
        if v is not None:
            key = k
            if hasattr(models.Task, key):
                query = query.filter(getattr(models.Task, key) == v)
    marker_task = None
    if marker is not None:
        marker_task = _task_get(context, session, marker, force_show_deleted=showing_deleted)
    sort_keys = ['created_at', 'id']
    if sort_key not in sort_keys:
        sort_keys.insert(0, sort_key)
    query = _paginate_query(query, models.Task, limit, sort_keys, marker=marker_task, sort_dir=sort_dir)
    task_refs = query.all()
    tasks = []
    for task_ref in task_refs:
        tasks.append(_task_format(task_ref, task_info_ref=None))
    return tasks