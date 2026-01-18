from unittest import mock
from oslo_db import exception as db_exc
import osprofiler
import sqlalchemy
from sqlalchemy.orm import exc
import testtools
from neutron_lib.db import api as db_api
from neutron_lib import exceptions
from neutron_lib import fixture
from neutron_lib.tests import _base
class TestDBProfiler(_base.BaseTestCase):

    @mock.patch.object(osprofiler.opts, 'is_trace_enabled', return_value=True)
    @mock.patch.object(osprofiler.opts, 'is_db_trace_enabled', return_value=True)
    def test_set_hook(self, _mock_dbt, _mock_t):
        with mock.patch.object(osprofiler.sqlalchemy, 'add_tracing') as add_tracing:
            engine_mock = mock.Mock()
            db_api._set_hook(engine_mock)
            add_tracing.assert_called_once_with(sqlalchemy, mock.ANY, 'neutron.db')