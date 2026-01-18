from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
from gslib.utils import boto_util
from gslib.utils import constants
from gslib.utils import system_util
from gslib.utils import ls_helper
from gslib.utils import retry_util
from gslib.utils import text_util
from gslib.utils import unit_util
import gslib.tests.testcase as testcase
from gslib.tests.util import SetEnvironmentForTest
from gslib.tests.util import TestParams
from gslib.utils.text_util import CompareVersions
from gslib.utils.unit_util import DecimalShort
from gslib.utils.unit_util import HumanReadableWithDecimalPlaces
from gslib.utils.unit_util import PrettyTime
import httplib2
import os
import six
from six import add_move, MovedModule
from six.moves import mock
def _AssertProxyInfosEqual(self, pi1, pi2):
    self.assertEqual(pi1.proxy_type, pi2.proxy_type)
    self.assertEqual(pi1.proxy_host, pi2.proxy_host)
    self.assertEqual(pi1.proxy_port, pi2.proxy_port)
    self.assertEqual(pi1.proxy_rdns, pi2.proxy_rdns)
    self.assertEqual(pi1.proxy_user, pi2.proxy_user)
    self.assertEqual(pi1.proxy_pass, pi2.proxy_pass)