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
def _is_nested_instance(e, etypes):
    """Check if exception or its inner excepts are an instance of etypes."""
    if isinstance(e, etypes):
        return True
    if isinstance(e, exceptions.MultipleExceptions):
        return any((_is_nested_instance(i, etypes) for i in e.inner_exceptions))
    if isinstance(e, db_exc.DBError):
        return _is_nested_instance(e.inner_exception, etypes)
    return False