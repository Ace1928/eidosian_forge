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
@_init_events.dispatch_for('sqlite')
def _init_events(engine, sqlite_synchronous=True, sqlite_fk=False, **kw):
    """Set up event listeners for SQLite.

    This includes several settings made on connections as they are
    created, as well as transactional control extensions.

    """

    def regexp(expr, item):
        reg = re.compile(expr)
        return reg.search(str(item)) is not None

    @sqlalchemy.event.listens_for(engine, 'connect')
    def _sqlite_connect_events(dbapi_con, con_record):
        dbapi_con.create_function('regexp', 2, regexp)
        if not sqlite_synchronous:
            dbapi_con.execute('PRAGMA synchronous = OFF')
        dbapi_con.isolation_level = None
        if sqlite_fk:
            dbapi_con.execute('pragma foreign_keys=ON')

    @sqlalchemy.event.listens_for(engine, 'begin')
    def _sqlite_emit_begin(conn):
        if 'in_transaction' not in conn.info:
            conn.execute(sqlalchemy.text('BEGIN'))
            conn.info['in_transaction'] = True

    @sqlalchemy.event.listens_for(engine, 'rollback')
    @sqlalchemy.event.listens_for(engine, 'commit')
    def _sqlite_end_transaction(conn):
        conn.info.pop('in_transaction', None)