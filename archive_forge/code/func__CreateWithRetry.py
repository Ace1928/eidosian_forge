from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import boto
import os
import re
from gslib.commands import hmac
from gslib.project_id import PopulateProjectId
import gslib.tests.testcase as testcase
from gslib.tests.testcase.integration_testcase import SkipForS3
from gslib.tests.testcase.integration_testcase import SkipForXML
from gslib.tests.util import SetBotoConfigForTest
from gslib.tests.util import SetEnvironmentForTest
from gslib.tests.util import unittest
from gslib.utils.retry_util import Retry
from gslib.utils import shim_util
from six import add_move, MovedModule
from six.moves import mock
@Retry(KeyLimitError, tries=5, timeout_secs=3)
def _CreateWithRetry(self, service_account):
    """Retry creation on key limit failures."""
    try:
        return self.RunGsUtil(['hmac', 'create', service_account], return_stdout=True)
    except AssertionError as e:
        if 'HMAC key limit reached' in str(e):
            raise KeyLimitError(str(e))
        else:
            raise