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
def _property_for_context(context):
    try:
        transaction_context = context.transaction_ctx
    except exception.NoEngineContextEstablished:
        raise exception.NoEngineContextEstablished('No TransactionContext is established for this %s object within the current thread; the %r attribute is unavailable.' % (context, attr))
    else:
        result = getter(transaction_context)
        if result is None:
            raise exception.ContextNotRequestedError("The '%s' context attribute was requested but it has not been established for this context." % attr)
        return result