import configparser
import logging
import logging.handlers
import os
import tempfile
from unittest import mock
import uuid
import fixtures
import testtools
from oslo_rootwrap import cmd
from oslo_rootwrap import daemon
from oslo_rootwrap import filters
from oslo_rootwrap import subprocess
from oslo_rootwrap import wrapper
class DaemonCleanupTestCase(testtools.TestCase):

    @mock.patch('os.chmod')
    @mock.patch('shutil.rmtree')
    @mock.patch('tempfile.mkdtemp')
    @mock.patch('multiprocessing.managers.BaseManager.get_server', side_effect=DaemonCleanupException)
    def test_daemon_no_cleanup_for_uninitialized_server(self, gs, mkd, *args):
        mkd.return_value = '/just_dir/123'
        self.assertRaises(DaemonCleanupException, daemon.daemon_start, config=None, filters=None)