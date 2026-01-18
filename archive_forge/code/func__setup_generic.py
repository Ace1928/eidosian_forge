from __future__ import absolute_import, print_function, division
import logging
from datetime import datetime, date
import sqlite3
import pytest
from petl.io.db import fromdb, todb
from petl.io.db_create import make_sqlalchemy_column
from petl.test.helpers import ieq, eq_
from petl.util.vis import look
from petl.test.io.test_db_server import user, password, host, database
def _setup_generic(dbapi_connection):
    cursor = dbapi_connection.cursor()
    cursor.execute('DROP TABLE IF EXISTS test_create')
    cursor.execute('DROP TABLE IF EXISTS "foo "" bar`"')
    cursor.close()
    dbapi_connection.commit()