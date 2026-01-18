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
class DialectFunctionDispatcher(object):

    @classmethod
    def dispatch_for_dialect(cls, expr, multiple=False):
        """Provide dialect-specific functionality within distinct functions.

        e.g.::

            @dispatch_for_dialect("*")
            def set_special_option(engine):
                pass

            @set_special_option.dispatch_for("sqlite")
            def set_sqlite_special_option(engine):
                return engine.execute("sqlite thing")

            @set_special_option.dispatch_for("mysql+mysqldb")
            def set_mysqldb_special_option(engine):
                return engine.execute("mysqldb thing")

        After the above registration, the ``set_special_option()`` function
        is now a dispatcher, given a SQLAlchemy ``Engine``, ``Connection``,
        URL string, or ``sqlalchemy.engine.URL`` object::

            eng = create_engine('...')
            result = set_special_option(eng)

        The filter system supports two modes, "multiple" and "single".
        The default is "single", and requires that one and only one function
        match for a given backend.    In this mode, the function may also
        have a return value, which will be returned by the top level
        call.

        "multiple" mode, on the other hand, does not support return
        arguments, but allows for any number of matching functions, where
        each function will be called::

            # the initial call sets this up as a "multiple" dispatcher
            @dispatch_for_dialect("*", multiple=True)
            def set_options(engine):
                # set options that apply to *all* engines

            @set_options.dispatch_for("postgresql")
            def set_postgresql_options(engine):
                # set options that apply to all Postgresql engines

            @set_options.dispatch_for("postgresql+psycopg2")
            def set_postgresql_psycopg2_options(engine):
                # set options that apply only to "postgresql+psycopg2"

            @set_options.dispatch_for("*+pyodbc")
            def set_pyodbc_options(engine):
                # set options that apply to all pyodbc backends

        Note that in both modes, any number of additional arguments can be
        accepted by member functions.  For example, to populate a dictionary of
        options, it may be passed in::

            @dispatch_for_dialect("*", multiple=True)
            def set_engine_options(url, opts):
                pass

            @set_engine_options.dispatch_for("mysql+mysqldb")
            def _mysql_set_default_charset_to_utf8(url, opts):
                opts.setdefault('charset', 'utf-8')

            @set_engine_options.dispatch_for("sqlite")
            def _set_sqlite_in_memory_check_same_thread(url, opts):
                if url.database in (None, 'memory'):
                    opts['check_same_thread'] = False

            opts = {}
            set_engine_options(url, opts)

        The driver specifiers are of the form:
        ``<database | *>[+<driver | *>]``.   That is, database name or "*",
        followed by an optional ``+`` sign with driver or "*".   Omitting
        the driver name implies all drivers for that database.

        """
        if multiple:
            cls = DialectMultiFunctionDispatcher
        else:
            cls = DialectSingleFunctionDispatcher
        return cls().dispatch_for(expr)
    _db_plus_driver_reg = re.compile('([^+]+?)(?:\\+(.+))?$')

    def dispatch_for(self, expr):

        def decorate(fn):
            dbname, driver = self._parse_dispatch(expr)
            if fn is self:
                fn = fn._last
            self._last = fn
            self._register(expr, dbname, driver, fn)
            return self
        return decorate

    def _parse_dispatch(self, text):
        m = self._db_plus_driver_reg.match(text)
        if not m:
            raise ValueError("Couldn't parse database[+driver]: %r" % text)
        return (m.group(1) or '*', m.group(2) or '*')

    def __call__(self, *arg, **kw):
        target = arg[0]
        return self._dispatch_on(self._url_from_target(target), target, arg, kw)

    def _url_from_target(self, target):
        if isinstance(target, Connectable):
            return target.engine.url
        elif isinstance(target, str):
            if '://' not in target:
                target_url = sa_url.make_url('%s://' % target)
            else:
                target_url = sa_url.make_url(target)
            return target_url
        elif isinstance(target, sa_url.URL):
            return target
        else:
            raise ValueError('Invalid target type: %r' % target)

    def dispatch_on_drivername(self, drivername):
        """Return a sub-dispatcher for the given drivername.

        This provides a means of calling a different function, such as the
        "*" function, for a given target object that normally refers
        to a sub-function.

        """
        dbname, driver = self._db_plus_driver_reg.match(drivername).group(1, 2)

        def go(*arg, **kw):
            return self._dispatch_on_db_driver(dbname, '*', arg, kw)
        return go

    def _dispatch_on(self, url, target, arg, kw):
        dbname, driver = self._db_plus_driver_reg.match(url.drivername).group(1, 2)
        if not driver:
            driver = url.get_dialect().driver
        return self._dispatch_on_db_driver(dbname, driver, arg, kw)

    def _invoke_fn(self, fn, arg, kw):
        return fn(*arg, **kw)