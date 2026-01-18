import collections
import logging
import re
import sys
from sqlalchemy import event
from sqlalchemy import exc as sqla_exc
from oslo_db import exception
from oslo_db.sqlalchemy import compat
@filters('mysql', sqla_exc.OperationalError, '^.*\\b1213\\b.*Deadlock found.*')
@filters('mysql', sqla_exc.DatabaseError, '^.*\\b1205\\b.*Lock wait timeout exceeded.*')
@filters('mysql', sqla_exc.InternalError, '^.*\\b1213\\b.*Deadlock found.*')
@filters('mysql', sqla_exc.InternalError, '^.*\\b1213\\b.*detected deadlock/conflict.*')
@filters('mysql', sqla_exc.InternalError, '^.*\\b1213\\b.*Deadlock: wsrep aborted.*')
@filters('postgresql', sqla_exc.OperationalError, '^.*deadlock detected.*')
@filters('postgresql', sqla_exc.DBAPIError, '^.*deadlock detected.*')
def _deadlock_error(operational_error, match, engine_name, is_disconnect):
    """Filter for MySQL or Postgresql deadlock error.

    NOTE(comstud): In current versions of DB backends, Deadlock violation
    messages follow the structure:

    mysql+mysqldb::

        (OperationalError) (1213, 'Deadlock found when trying to get lock; '
            'try restarting transaction') <query_str> <query_args>

    mysql+mysqlconnector::

        (InternalError) 1213 (40001): Deadlock found when trying to get lock;
            try restarting transaction

    postgresql::

        (TransactionRollbackError) deadlock detected <deadlock_details>
    """
    raise exception.DBDeadlock(operational_error)