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
class QueryParamTest(db_test_base._DbTestCase):

    def _fixture(self):
        from sqlalchemy import create_engine

        def _mock_create_engine(*arg, **kw):
            return create_engine('sqlite://')
        return mock.patch('oslo_db.sqlalchemy.engines.sqlalchemy.create_engine', side_effect=_mock_create_engine)

    def _normalize_query_dict(self, qdict):
        return {k: list(v) if isinstance(v, tuple) else v for k, v in qdict.items()}

    def test_add_assorted_params(self):
        with self._fixture() as ce:
            engines.create_engine('mysql+pymysql://foo:bar@bat', connection_parameters='foo=bar&bat=hoho&bat=param2')
        self.assertEqual(self._normalize_query_dict(ce.mock_calls[0][1][0].query), {'bat': ['hoho', 'param2'], 'foo': 'bar'})

    def test_add_no_params(self):
        with self._fixture() as ce:
            engines.create_engine('mysql+pymysql://foo:bar@bat')
        self.assertEqual(ce.mock_calls[0][1][0].query, self._normalize_query_dict({}))

    def test_combine_params(self):
        with self._fixture() as ce:
            engines.create_engine('mysql+pymysql://foo:bar@bat/?charset=utf8&param_file=tripleo.cnf', connection_parameters='plugin=sqlalchemy_collectd&collectd_host=127.0.0.1&bind_host=192.168.1.5')
        self.assertEqual(self._normalize_query_dict(ce.mock_calls[0][1][0].query), {'bind_host': '192.168.1.5', 'charset': 'utf8', 'collectd_host': '127.0.0.1', 'param_file': 'tripleo.cnf', 'plugin': 'sqlalchemy_collectd'})

    def test_combine_multi_params(self):
        with self._fixture() as ce:
            engines.create_engine('mysql+pymysql://foo:bar@bat/?charset=utf8&param_file=tripleo.cnf&plugin=connmon', connection_parameters='plugin=sqlalchemy_collectd&collectd_host=127.0.0.1&bind_host=192.168.1.5')
        self.assertEqual(self._normalize_query_dict(ce.mock_calls[0][1][0].query), {'bind_host': '192.168.1.5', 'charset': 'utf8', 'collectd_host': '127.0.0.1', 'param_file': 'tripleo.cnf', 'plugin': ['connmon', 'sqlalchemy_collectd']})