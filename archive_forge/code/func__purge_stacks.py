import datetime
import functools
import itertools
import random
from oslo_config import cfg
from oslo_db import api as oslo_db_api
from oslo_db import exception as db_exception
from oslo_db import options
from oslo_db.sqlalchemy import enginefacade
from oslo_db.sqlalchemy import utils
from oslo_log import log as logging
from oslo_utils import encodeutils
from oslo_utils import timeutils
import sqlalchemy
from sqlalchemy import and_
from sqlalchemy import func
from sqlalchemy import or_
from sqlalchemy import orm
from heat.common import crypt
from heat.common import exception
from heat.common.i18n import _
from heat.db import filters as db_filters
from heat.db import models
from heat.db import utils as db_utils
from heat.engine import environment as heat_environment
from heat.rpc import api as rpc_api
@oslo_db_api.wrap_db_retry(max_retries=3, retry_on_deadlock=True, retry_interval=0.5, inc_retry_interval=True)
def _purge_stacks(stack_infos, engine, meta):
    """Purge some stacks and their releated events, raw_templates, etc.

    stack_infos is a list of lists of selected stack columns:
    [[id, raw_template_id, prev_raw_template_id, user_creds_id,
      action, status, name], ...]
    """
    with engine.connect() as conn, conn.begin():
        stack = sqlalchemy.Table('stack', meta, autoload_with=conn)
        stack_lock = sqlalchemy.Table('stack_lock', meta, autoload_with=conn)
        stack_tag = sqlalchemy.Table('stack_tag', meta, autoload_with=conn)
        resource = sqlalchemy.Table('resource', meta, autoload_with=conn)
        resource_data = sqlalchemy.Table('resource_data', meta, autoload_with=conn)
        resource_properties_data = sqlalchemy.Table('resource_properties_data', meta, autoload_with=conn)
        event = sqlalchemy.Table('event', meta, autoload_with=conn)
        raw_template = sqlalchemy.Table('raw_template', meta, autoload_with=conn)
        raw_template_files = sqlalchemy.Table('raw_template_files', meta, autoload_with=conn)
        user_creds = sqlalchemy.Table('user_creds', meta, autoload_with=conn)
        syncpoint = sqlalchemy.Table('sync_point', meta, autoload_with=conn)
    stack_info_str = ','.join([str(i) for i in stack_infos])
    LOG.info('Purging stacks %s', stack_info_str)
    stack_ids = [stack_info[0] for stack_info in stack_infos]
    stack_lock_del = stack_lock.delete().where(stack_lock.c.stack_id.in_(stack_ids))
    with engine.connect() as conn, conn.begin():
        conn.execute(stack_lock_del)
    stack_tag_del = stack_tag.delete().where(stack_tag.c.stack_id.in_(stack_ids))
    with engine.connect() as conn, conn.begin():
        conn.execute(stack_tag_del)
    res_where = sqlalchemy.select(resource.c.id).where(resource.c.stack_id.in_(stack_ids))
    res_data_del = resource_data.delete().where(resource_data.c.resource_id.in_(res_where))
    with engine.connect() as conn, conn.begin():
        conn.execute(res_data_del)
    sync_del = syncpoint.delete().where(syncpoint.c.stack_id.in_(stack_ids))
    with engine.connect() as conn, conn.begin():
        conn.execute(sync_del)
    rsrc_prop_data_where = sqlalchemy.select(resource.c.rsrc_prop_data_id).where(resource.c.stack_id.in_(stack_ids))
    with engine.connect() as conn, conn.begin():
        rsrc_prop_data_ids = set([i[0] for i in list(conn.execute(rsrc_prop_data_where))])
    rsrc_prop_data_where = sqlalchemy.select(resource.c.attr_data_id).where(resource.c.stack_id.in_(stack_ids))
    with engine.connect() as conn, conn.begin():
        rsrc_prop_data_ids.update([i[0] for i in list(conn.execute(rsrc_prop_data_where))])
    rsrc_prop_data_where = sqlalchemy.select(event.c.rsrc_prop_data_id).where(event.c.stack_id.in_(stack_ids))
    with engine.connect() as conn, conn.begin():
        rsrc_prop_data_ids.update([i[0] for i in list(conn.execute(rsrc_prop_data_where))])
    event_del = event.delete().where(event.c.stack_id.in_(stack_ids))
    with engine.connect() as conn, conn.begin():
        conn.execute(event_del)
    res_del = resource.delete().where(resource.c.stack_id.in_(stack_ids))
    with engine.connect() as conn, conn.begin():
        conn.execute(res_del)
    if rsrc_prop_data_ids:
        rsrc_prop_data_where = sqlalchemy.select(event.c.rsrc_prop_data_id).where(event.c.rsrc_prop_data_id.in_(rsrc_prop_data_ids))
        with engine.connect() as conn, conn.begin():
            ids = list(conn.execute(rsrc_prop_data_where))
        rsrc_prop_data_ids.difference_update([i[0] for i in ids])
    if rsrc_prop_data_ids:
        rsrc_prop_data_where = sqlalchemy.select(resource.c.rsrc_prop_data_id).where(resource.c.rsrc_prop_data_id.in_(rsrc_prop_data_ids))
        with engine.connect() as conn, conn.begin():
            ids = list(conn.execute(rsrc_prop_data_where))
        rsrc_prop_data_ids.difference_update([i[0] for i in ids])
    if rsrc_prop_data_ids:
        rsrc_prop_data_del = resource_properties_data.delete().where(resource_properties_data.c.id.in_(rsrc_prop_data_ids))
        with engine.connect() as conn, conn.begin():
            conn.execute(rsrc_prop_data_del)
    stack_del = stack.delete().where(stack.c.id.in_(stack_ids))
    with engine.connect() as conn, conn.begin():
        conn.execute(stack_del)
    raw_template_ids = [i[1] for i in stack_infos if i[1] is not None]
    raw_template_ids.extend((i[2] for i in stack_infos if i[2] is not None))
    if raw_template_ids:
        raw_tmpl_sel = sqlalchemy.select(stack.c.raw_template_id).where(stack.c.raw_template_id.in_(raw_template_ids))
        with engine.connect() as conn, conn.begin():
            raw_tmpl = [i[0] for i in conn.execute(raw_tmpl_sel)]
        raw_template_ids = set(raw_template_ids) - set(raw_tmpl)
    if raw_template_ids:
        raw_tmpl_sel = sqlalchemy.select(stack.c.prev_raw_template_id).where(stack.c.prev_raw_template_id.in_(raw_template_ids))
        with engine.connect() as conn, conn.begin():
            raw_tmpl = [i[0] for i in conn.execute(raw_tmpl_sel)]
        raw_template_ids = raw_template_ids - set(raw_tmpl)
    if raw_template_ids:
        raw_tmpl_file_sel = sqlalchemy.select(raw_template.c.files_id).where(raw_template.c.id.in_(raw_template_ids))
        with engine.connect() as conn, conn.begin():
            raw_tmpl_file_ids = [i[0] for i in conn.execute(raw_tmpl_file_sel)]
        raw_templ_del = raw_template.delete().where(raw_template.c.id.in_(raw_template_ids))
        with engine.connect() as conn, conn.begin():
            conn.execute(raw_templ_del)
        if raw_tmpl_file_ids:
            raw_tmpl_file_sel = sqlalchemy.select(raw_template.c.files_id).where(raw_template.c.files_id.in_(raw_tmpl_file_ids))
            with engine.connect() as conn, conn.begin():
                raw_tmpl_files = [i[0] for i in conn.execute(raw_tmpl_file_sel)]
            raw_tmpl_file_ids = set(raw_tmpl_file_ids) - set(raw_tmpl_files)
        if raw_tmpl_file_ids:
            raw_tmpl_file_del = raw_template_files.delete().where(raw_template_files.c.id.in_(raw_tmpl_file_ids))
            with engine.connect() as conn, conn.begin():
                conn.execute(raw_tmpl_file_del)
    user_creds_ids = [i[3] for i in stack_infos if i[3] is not None]
    if user_creds_ids:
        user_sel = sqlalchemy.select(stack.c.user_creds_id).where(stack.c.user_creds_id.in_(user_creds_ids))
        with engine.connect() as conn, conn.begin():
            users = [i[0] for i in conn.execute(user_sel)]
        user_creds_ids = set(user_creds_ids) - set(users)
    if user_creds_ids:
        usr_creds_del = user_creds.delete().where(user_creds.c.id.in_(user_creds_ids))
        with engine.connect() as conn, conn.begin():
            conn.execute(usr_creds_del)