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
def AssertKeyMetadataMatches(self, output_string, access_id='GOOG.*', state='ACTIVE', service_account='.*', project='.*'):
    self.assertRegex(output_string, 'Access ID %s:' % access_id)
    self.assertRegex(output_string, '\\sState:\\s+%s' % state)
    self.assertRegex(output_string, '\\s+Service Account:\\s+%s\\n' % service_account)
    self.assertRegex(output_string, '\\s+Project:\\s+%s' % project)
    self.assertRegex(output_string, '\\s+Time Created:\\s+.*')
    self.assertRegex(output_string, '\\s+Time Last Updated:\\s+.*')