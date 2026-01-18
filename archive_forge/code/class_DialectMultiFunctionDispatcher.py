import collections
from collections import abc
import itertools
import logging
import re
from oslo_utils import timeutils
import sqlalchemy
from sqlalchemy import Boolean
from sqlalchemy.engine import Connectable
from sqlalchemy.engine import url as sa_url
from sqlalchemy import exc
from sqlalchemy import func
from sqlalchemy import Index
from sqlalchemy import inspect
from sqlalchemy import Integer
from sqlalchemy import MetaData
from sqlalchemy.sql.expression import cast
from sqlalchemy.sql.expression import literal_column
from sqlalchemy.sql import text
from sqlalchemy import Table
from oslo_db._i18n import _
from oslo_db import exception
from oslo_db.sqlalchemy import models
class DialectMultiFunctionDispatcher(DialectFunctionDispatcher):

    def __init__(self):
        self.reg = collections.defaultdict(lambda: collections.defaultdict(list))

    def _register(self, expr, dbname, driver, fn):
        self.reg[dbname][driver].append(fn)

    def _matches(self, dbname, driver):
        if driver != '*':
            drivers = (driver, '*')
        else:
            drivers = ('*',)
        for db in (dbname, '*'):
            subdict = self.reg[db]
            for drv in drivers:
                for fn in subdict[drv]:
                    yield fn

    def _dispatch_on_db_driver(self, dbname, driver, arg, kw):
        for fn in self._matches(dbname, driver):
            if self._invoke_fn(fn, arg, kw) is not None:
                raise TypeError('Return value not allowed for multiple filtered function')