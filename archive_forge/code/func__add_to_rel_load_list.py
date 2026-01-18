import contextlib
import copy
import functools
import weakref
from oslo_concurrency import lockutils
from oslo_config import cfg
from oslo_db import api as oslo_db_api
from oslo_db import exception as db_exc
from oslo_db.sqlalchemy import enginefacade
from oslo_log import log as logging
from oslo_utils import excutils
from osprofiler import opts as profiler_opts
import osprofiler.sqlalchemy
from pecan import util as p_util
import sqlalchemy
from sqlalchemy import event  # noqa
from sqlalchemy import exc as sql_exc
from sqlalchemy import orm
from sqlalchemy.orm import exc
from neutron_lib._i18n import _
from neutron_lib.db import model_base
from neutron_lib import exceptions
from neutron_lib.objects import exceptions as obj_exc
@event.listens_for(orm.session.Session, 'after_flush')
def _add_to_rel_load_list(session, flush_context=None):
    session.info.setdefault('_load_rels', weakref.WeakSet()).update(session.new)