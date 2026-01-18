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
def _filter_and_page_query(context, query, limit=None, sort_keys=None, marker=None, sort_dir=None, filters=None):
    if filters is None:
        filters = {}
    sort_key_map = {rpc_api.STACK_NAME: models.Stack.name.key, rpc_api.STACK_STATUS: models.Stack.status.key, rpc_api.STACK_CREATION_TIME: models.Stack.created_at.key, rpc_api.STACK_UPDATED_TIME: models.Stack.updated_at.key}
    valid_sort_keys = _get_sort_keys(sort_keys, sort_key_map)
    query = db_filters.exact_filter(query, models.Stack, filters)
    return _paginate_query(context, query, models.Stack, limit, valid_sort_keys, marker, sort_dir)