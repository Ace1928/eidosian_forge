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
@BackendImpl.impl.dispatch_for('sqlite')
class SQLiteBackendImpl(BackendImpl):
    supports_drop_fk = False

    def dispose(self, engine):
        LOG.info('DISPOSE ENGINE %s', engine)
        engine.dispose()
        url = engine.url
        self._drop_url_file(url, True)

    def _drop_url_file(self, url, conditional):
        filename = url.database
        if filename and (not conditional or os.access(filename, os.F_OK)):
            os.remove(filename)

    def create_opportunistic_driver_url(self):
        return 'sqlite://'

    def create_named_database(self, engine, ident, conditional=False):
        url = self.provisioned_database_url(engine.url, ident)
        filename = url.database
        if filename and (not conditional or not os.access(filename, os.F_OK)):
            eng = sqlalchemy.create_engine(url)
            eng.connect().close()

    def drop_named_database(self, engine, ident, conditional=False):
        url = self.provisioned_database_url(engine.url, ident)
        filename = url.database
        if filename and (not conditional or os.access(filename, os.F_OK)):
            os.remove(filename)

    def database_exists(self, engine, ident):
        url = self._provisioned_database_url(engine.url, ident)
        filename = url.database
        return not filename or os.access(filename, os.F_OK)

    def provisioned_database_url(self, base_url, ident):
        if base_url.database:
            return utils.make_url('sqlite:////tmp/%s.db' % ident)
        else:
            return base_url