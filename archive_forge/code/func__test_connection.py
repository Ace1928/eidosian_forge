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
def _test_connection(engine, max_retries, retry_interval):
    if max_retries == -1:
        attempts = itertools.count()
    else:
        attempts = range(max_retries)
    de_ref = None
    for attempt in attempts:
        try:
            return engine.connect()
        except exception.DBConnectionError as de:
            msg = 'SQL connection failed. %s attempts left.'
            LOG.warning(msg, max_retries - attempt)
            time.sleep(retry_interval)
            de_ref = de
    else:
        if de_ref is not None:
            raise de_ref