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
def _all_backup_stack_ids(context, stack_id):
    """Iterate over all the IDs of all stacks related as stack/backup pairs.

    All backup stacks of a main stack, past and present (i.e. including those
    that are soft deleted), are included. The main stack itself is also
    included if the initial ID passed in is for a backup stack. The initial ID
    passed in is never included in the output.
    """
    stack = context.session.get(models.Stack, stack_id)
    if stack is None:
        LOG.error('Stack %s not found', stack_id)
        return
    is_backup = stack.name.endswith('*')
    if is_backup:
        main = context.session.get(models.Stack, stack.owner_id)
        if main is None:
            LOG.error('Main stack for backup "%s" %s not found', stack.name, stack_id)
            return
        yield main.id
        for backup_id in _all_backup_stack_ids(context, main.id):
            if backup_id != stack_id:
                yield backup_id
    else:
        q_backup = context.session.query(models.Stack).filter(sqlalchemy.or_(models.Stack.tenant == context.tenant_id, models.Stack.stack_user_project_id == context.tenant_id))
        q_backup = q_backup.filter_by(name=stack.name + '*')
        q_backup = q_backup.filter_by(owner_id=stack_id)
        for backup in q_backup.all():
            yield backup.id