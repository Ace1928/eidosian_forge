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
def _events_paginate_query(context, query, model, limit=None, sort_keys=None, marker=None, sort_dir=None):
    default_sort_keys = ['created_at']
    if not sort_keys:
        sort_keys = default_sort_keys
        if not sort_dir:
            sort_dir = 'desc'
    sort_keys = sort_keys + ['id']
    model_marker = None
    if marker:
        model_marker = context.session.query(model).filter_by(uuid=marker).first()
    try:
        query = utils.paginate_query(query, model, limit, sort_keys, model_marker, sort_dir)
    except utils.InvalidSortKey as exc:
        err_msg = encodeutils.exception_to_unicode(exc)
        raise exception.Invalid(reason=err_msg)
    return query