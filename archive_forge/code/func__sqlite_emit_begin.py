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
@sqlalchemy.event.listens_for(engine, 'begin')
def _sqlite_emit_begin(conn):
    if 'in_transaction' not in conn.info:
        conn.execute(sqlalchemy.text('BEGIN'))
        conn.info['in_transaction'] = True