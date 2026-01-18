import copy
import os
import sys
from unittest import mock
import testtools
from osc_lib import shell
from osc_lib.tests import utils
class TestShellArgV(utils.TestShell):
    """Test the deferred help flag"""

    def setUp(self):
        super(TestShellArgV, self).setUp()

    def test_shell_argv(self):
        """Test argv decoding

        Python 2 does nothing with argv while Python 3 decodes it into
        Unicode before we ever see it.  We manually decode when running
        under Python 2 so verify that we get the right argv types.

        Use the argv supplied by the test runner so we get actual Python
        runtime behaviour; we only need to check the type of argv[0]
        which will alwyas be present.
        """
        with mock.patch('osc_lib.shell.OpenStackShell.run', self.app):
            argv = sys.argv
            shell.main(sys.argv)
            self.assertEqual(type(argv[0]), type(self.app.call_args[0][0][0]))
            shell.main()
            self.assertEqual(type(u'x'), type(self.app.call_args[0][0][0]))