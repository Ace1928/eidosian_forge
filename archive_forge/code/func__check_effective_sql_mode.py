import functools
import itertools
import logging
import os
import re
import time
import debtcollector.removals
import debtcollector.renames
import sqlalchemy
from sqlalchemy import event
from sqlalchemy import exc
from sqlalchemy import pool
from sqlalchemy import select
from oslo_db import exception
from oslo_db.sqlalchemy import compat
from oslo_db.sqlalchemy import exc_filters
from oslo_db.sqlalchemy import utils
@sqlalchemy.event.listens_for(engine, 'first_connect')
def _check_effective_sql_mode(dbapi_con, connection_rec):
    if mysql_sql_mode is not None or mysql_wsrep_sync_wait is not None:
        _set_session_variables(dbapi_con, connection_rec)
    cursor = dbapi_con.cursor()
    cursor.execute("SHOW VARIABLES LIKE 'sql_mode'")
    realmode = cursor.fetchone()
    if realmode is None:
        LOG.warning('Unable to detect effective SQL mode')
    else:
        realmode = realmode[1]
        LOG.debug('MySQL server mode set to %s', realmode)
        if 'TRADITIONAL' not in realmode.upper() and 'STRICT_ALL_TABLES' not in realmode.upper():
            LOG.warning("MySQL SQL mode is '%s', consider enabling TRADITIONAL or STRICT_ALL_TABLES", realmode)