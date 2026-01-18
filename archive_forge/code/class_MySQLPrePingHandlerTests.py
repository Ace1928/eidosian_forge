import contextlib
import itertools
from unittest import mock
import sqlalchemy as sqla
from sqlalchemy import event
import sqlalchemy.exc
from sqlalchemy.orm import declarative_base
from sqlalchemy.orm import registry
from sqlalchemy import sql
from oslo_db import exception
from oslo_db.sqlalchemy import compat
from oslo_db.sqlalchemy import engines
from oslo_db.sqlalchemy import exc_filters
from oslo_db.sqlalchemy import utils
from oslo_db.tests import base as test_base
from oslo_db.tests.sqlalchemy import base as db_test_base
from oslo_db.tests import utils as test_utils
class MySQLPrePingHandlerTests(db_test_base._MySQLOpportunisticTestCase, TestDBDisconnectedFixture):

    def test_mariadb_error_1927(self):
        for code in [1927]:
            self._test_ping_listener_disconnected('mysql', self.InternalError('%d Connection was killed' % code), is_disconnect=False)

    def test_packet_sequence_wrong_error(self):
        self._test_ping_listener_disconnected('mysql', self.InternalError('Packet sequence number wrong - got 35 expected 1'), is_disconnect=False)

    def test_mysql_ping_listener_disconnected(self):
        for code in [2006, 2013, 2014, 2045, 2055]:
            self._test_ping_listener_disconnected('mysql', self.OperationalError('%d MySQL server has gone away' % code))

    def test_mysql_ping_listener_disconnected_regex_only(self):
        for code in [2002, 2003, 2006, 2013]:
            self._test_ping_listener_disconnected('mysql', self.OperationalError('%d MySQL server has gone away' % code), is_disconnect=False)

    def test_mysql_galera_non_primary_disconnected(self):
        self._test_ping_listener_disconnected('mysql', self.OperationalError("(1047, 'Unknown command') 'SELECT DATABASE()' ()"))

    def test_mysql_galera_non_primary_disconnected_regex_only(self):
        self._test_ping_listener_disconnected('mysql', self.OperationalError("(1047, 'Unknown command') 'SELECT DATABASE()' ()"), is_disconnect=False)

    def test_mysql_w_disconnect_flag(self):
        for code in [2002, 2003, 2002]:
            self._test_ping_listener_disconnected('mysql', self.OperationalError('%d MySQL server has gone away' % code))

    def test_mysql_wo_disconnect_flag(self):
        for code in [2002, 2003]:
            self._test_ping_listener_disconnected('mysql', self.OperationalError('%d MySQL server has gone away' % code), is_disconnect=False)