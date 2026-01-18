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
def _sqlite_connect_events(dbapi_con, con_record):
    dbapi_con.create_function('regexp', 2, regexp)
    if not sqlite_synchronous:
        dbapi_con.execute('PRAGMA synchronous = OFF')
    dbapi_con.isolation_level = None
    if sqlite_fk:
        dbapi_con.execute('pragma foreign_keys=ON')