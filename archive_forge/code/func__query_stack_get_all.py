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
def _query_stack_get_all(context, show_deleted=False, show_nested=False, show_hidden=False, tags=None, tags_any=None, not_tags=None, not_tags_any=None):
    if show_nested:
        query = _soft_delete_aware_query(context, models.Stack, show_deleted=show_deleted).filter_by(backup=False)
    else:
        query = _soft_delete_aware_query(context, models.Stack, show_deleted=show_deleted).filter_by(owner_id=None)
    if not context.is_admin:
        query = query.filter_by(tenant=context.tenant_id)
    query = query.options(orm.subqueryload(models.Stack.tags))
    if tags:
        for tag in tags:
            tag_alias = orm.aliased(models.StackTag)
            query = query.join(tag_alias, models.Stack.tags)
            query = query.filter(tag_alias.tag == tag)
    if tags_any:
        query = query.filter(models.Stack.tags.any(models.StackTag.tag.in_(tags_any)))
    if not_tags:
        subquery = _soft_delete_aware_query(context, models.Stack, show_deleted=show_deleted)
        for tag in not_tags:
            tag_alias = orm.aliased(models.StackTag)
            subquery = subquery.join(tag_alias, models.Stack.tags)
            subquery = subquery.filter(tag_alias.tag == tag)
        not_stack_ids = [s.id for s in subquery.all()]
        query = query.filter(models.Stack.id.notin_(not_stack_ids))
    if not_tags_any:
        query = query.filter(~models.Stack.tags.any(models.StackTag.tag.in_(not_tags_any)))
    if not show_hidden and cfg.CONF.hidden_stack_tags:
        query = query.filter(~models.Stack.tags.any(models.StackTag.tag.in_(cfg.CONF.hidden_stack_tags)))
    return query