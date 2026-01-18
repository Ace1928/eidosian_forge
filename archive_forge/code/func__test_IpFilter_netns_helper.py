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
def _test_IpFilter_netns_helper(self, action):
    f = filters.IpFilter(self._ip, 'root')
    self.assertTrue(f.match(['ip', 'link', action]))