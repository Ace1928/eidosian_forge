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
class DialectSingleFunctionDispatcher(DialectFunctionDispatcher):

    def __init__(self):
        self.reg = collections.defaultdict(dict)

    def _register(self, expr, dbname, driver, fn):
        fn_dict = self.reg[dbname]
        if driver in fn_dict:
            raise TypeError('Multiple functions for expression %r' % expr)
        fn_dict[driver] = fn

    def _matches(self, dbname, driver):
        for db in (dbname, '*'):
            subdict = self.reg[db]
            for drv in (driver, '*'):
                if drv in subdict:
                    return subdict[drv]
        else:
            raise ValueError('No default function found for driver: %r' % ('%s+%s' % (dbname, driver)))

    def _dispatch_on_db_driver(self, dbname, driver, arg, kw):
        fn = self._matches(dbname, driver)
        return self._invoke_fn(fn, arg, kw)