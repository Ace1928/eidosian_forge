import testscenarios
from unittest import mock
from cliff import command
from cliff import commandmanager
from cliff.tests import base
from cliff.tests import utils
class TestLookupWithRemainder(base.TestBase):
    scenarios = [('one', {'argv': ['one', '--opt']}), ('two', {'argv': ['two', 'words', '--opt']}), ('three', {'argv': ['three', 'word', 'command', '--opt']})]

    def test(self):
        mgr = utils.TestCommandManager(utils.TEST_NAMESPACE)
        cmd, name, remaining = mgr.find_command(self.argv)
        self.assertTrue(cmd)
        self.assertEqual(['--opt'], remaining)