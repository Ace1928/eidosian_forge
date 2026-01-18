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
def _transaction_contexts_by_thread(context):
    transaction_contexts_by_thread = getattr(context, '_enginefacade_context', None)
    if transaction_contexts_by_thread is None:
        transaction_contexts_by_thread = context._enginefacade_context = _TransactionContextTLocal()
    return transaction_contexts_by_thread