import abc
import logging
import os
import random
import re
import string
import sqlalchemy
from sqlalchemy import schema
from sqlalchemy import sql
import testresources
from oslo_db import exception
from oslo_db.sqlalchemy import enginefacade
from oslo_db.sqlalchemy import utils
@BackendImpl.impl.dispatch_for('mysql')
class MySQLBackendImpl(BackendImpl):

    def create_opportunistic_driver_url(self):
        return 'mysql+pymysql://openstack_citest:openstack_citest@localhost/'

    def create_named_database(self, engine, ident, conditional=False):
        with engine.begin() as conn:
            if not conditional or not self.database_exists(conn, ident):
                conn.exec_driver_sql('CREATE DATABASE %s' % ident)

    def drop_named_database(self, engine, ident, conditional=False):
        with engine.begin() as conn:
            if not conditional or self.database_exists(conn, ident):
                conn.exec_driver_sql('DROP DATABASE %s' % ident)

    def database_exists(self, engine, ident):
        s = sql.text('SHOW DATABASES LIKE :ident')
        return bool(engine.scalar(s, {'ident': ident}))