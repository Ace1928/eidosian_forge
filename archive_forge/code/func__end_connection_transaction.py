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
def _end_connection_transaction(self, transaction):
    if self.mode is _WRITER:
        transaction.commit()
    else:
        transaction.rollback()