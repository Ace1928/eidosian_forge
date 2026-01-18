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
def _add_process_guards(engine):
    """Add multiprocessing guards.

    Forces a connection to be reconnected if it is detected
    as having been shared to a sub-process.

    """

    @sqlalchemy.event.listens_for(engine, 'connect')
    def connect(dbapi_connection, connection_record):
        connection_record.info['pid'] = os.getpid()

    @sqlalchemy.event.listens_for(engine, 'checkout')
    def checkout(dbapi_connection, connection_record, connection_proxy):
        pid = os.getpid()
        if connection_record.info['pid'] != pid:
            LOG.debug('Parent process %(orig)s forked (%(newproc)s) with an open database connection, which is being discarded and recreated.', {'newproc': pid, 'orig': connection_record.info['pid']})
            raise exc.DisconnectionError('Connection record belongs to pid %s, attempting to check out in pid %s' % (connection_record.info['pid'], pid))