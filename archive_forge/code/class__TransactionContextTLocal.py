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
class _TransactionContextTLocal(threading.local):

    def __deepcopy__(self, memo):
        return self

    def __reduce__(self):
        return (_TransactionContextTLocal, ())