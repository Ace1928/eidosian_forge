import contextlib
import functools
import inspect
import operator
import threading
import warnings
import debtcollector.moves
import debtcollector.removals
import debtcollector.renames
from oslo_config import cfg
from oslo_utils import excutils
from oslo_db import exception
from oslo_db import options
from oslo_db.sqlalchemy import engines
from oslo_db.sqlalchemy import orm
from oslo_db import warning
def _transaction_ctx_for_context(context):
    by_thread = _transaction_contexts_by_thread(context)
    try:
        return by_thread.current
    except AttributeError:
        raise exception.NoEngineContextEstablished('No TransactionContext is established for this %s object within the current thread. ' % context)