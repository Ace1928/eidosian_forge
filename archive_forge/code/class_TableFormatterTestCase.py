from unittest import mock
import testtools
from blazarclient import command
from blazarclient import tests
class TableFormatterTestCase(tests.TestCase):

    def setUp(self):
        super(TableFormatterTestCase, self).setUp()

    @testtools.skip('Have no idea how to test super')
    def test_emit_list(self):
        pass