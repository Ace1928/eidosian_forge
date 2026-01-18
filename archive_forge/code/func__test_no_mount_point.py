import os
import tempfile
from unittest import mock
from oslo_concurrency import processutils as putils
from os_brick import exception
from os_brick.privileged import rootwrap as priv_rootwrap
from os_brick.remotefs import remotefs
from os_brick.tests import base
def _test_no_mount_point(self, fs_type):
    self.assertRaises(exception.InvalidParameterValue, remotefs.RemoteFsClient, fs_type, root_helper='true')