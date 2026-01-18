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
@_init_connection_args.dispatch_for('mysql+mysqldb')
def _init_connection_args(url, engine_args, kw):
    if 'use_unicode' not in url.query:
        engine_args['connect_args']['use_unicode'] = 1