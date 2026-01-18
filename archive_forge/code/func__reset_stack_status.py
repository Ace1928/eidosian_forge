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
def _reset_stack_status(context, stack_id, stack=None):
    if stack is None:
        stack = context.session.get(models.Stack, stack_id)
    if stack is None:
        raise exception.NotFound(_('Stack with id %s not found') % stack_id)
    query = context.session.query(models.Resource).filter_by(status='IN_PROGRESS', stack_id=stack_id)
    query.update({'status': 'FAILED', 'status_reason': 'Stack status manually reset', 'engine_id': None})
    query = context.session.query(models.ResourceData)
    query = query.join(models.Resource)
    query = query.filter_by(stack_id=stack_id)
    query = query.filter(models.ResourceData.key.in_(heat_environment.HOOK_TYPES))
    data_ids = [data.id for data in query]
    if data_ids:
        query = context.session.query(models.ResourceData)
        query = query.filter(models.ResourceData.id.in_(data_ids))
        query.delete(synchronize_session='fetch')
    context.session.commit()
    query = context.session.query(models.Stack).filter_by(owner_id=stack_id)
    for child in query:
        _reset_stack_status(context, child.id, child)
    if stack.status == 'IN_PROGRESS':
        stack.status = 'FAILED'
        stack.status_reason = 'Stack status manually reset'
    context.session.query(models.StackLock).filter_by(stack_id=stack_id).delete()