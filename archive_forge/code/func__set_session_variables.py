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
@sqlalchemy.event.listens_for(engine, 'connect')
def _set_session_variables(dbapi_con, connection_rec):
    cursor = dbapi_con.cursor()
    if mysql_sql_mode is not None:
        cursor.execute('SET SESSION sql_mode = %s', [mysql_sql_mode])
    if mysql_wsrep_sync_wait is not None:
        cursor.execute('SET SESSION wsrep_sync_wait = %s', [mysql_wsrep_sync_wait])