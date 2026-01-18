import logging
import os
from unittest import mock
import fixtures
from oslo_config import cfg
import sqlalchemy
from sqlalchemy.engine import base as base_engine
from sqlalchemy import exc
from sqlalchemy.pool import NullPool
from sqlalchemy import sql
from sqlalchemy import Column, MetaData, Table
from sqlalchemy import Integer, String
from sqlalchemy.orm import declarative_base
from oslo_db import exception
from oslo_db import options as db_options
from oslo_db.sqlalchemy import compat
from oslo_db.sqlalchemy import enginefacade
from oslo_db.sqlalchemy import engines
from oslo_db.sqlalchemy import models
from oslo_db.sqlalchemy import session
from oslo_db.sqlalchemy import utils
from oslo_db.tests import base as test_base
from oslo_db.tests.sqlalchemy import base as db_test_base
class RegexpFilterTestCase(db_test_base._DbTestCase):

    def setUp(self):
        super(RegexpFilterTestCase, self).setUp()
        meta = MetaData()
        test_table = Table(_REGEXP_TABLE_NAME, meta, Column('id', Integer, primary_key=True, nullable=False), Column('bar', String(255)))
        test_table.create(self.engine)
        self.addCleanup(test_table.drop, self.engine)

    def _test_regexp_filter(self, regexp, expected):
        with enginefacade.writer.using(db_test_base.context):
            _session = db_test_base.context.session
            for i in ['10', '20', '♥']:
                tbl = RegexpTable()
                tbl.update({'bar': i})
                tbl.save(session=_session)
        regexp_op = RegexpTable.bar.op('REGEXP')(regexp)
        result = _session.query(RegexpTable).filter(regexp_op).all()
        self.assertEqual(expected, [r.bar for r in result])

    def test_regexp_filter(self):
        self._test_regexp_filter('10', ['10'])

    def test_regexp_filter_nomatch(self):
        self._test_regexp_filter('11', [])

    def test_regexp_filter_unicode(self):
        self._test_regexp_filter('♥', ['♥'])

    def test_regexp_filter_unicode_nomatch(self):
        self._test_regexp_filter('♦', [])