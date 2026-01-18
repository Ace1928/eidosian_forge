from __future__ import absolute_import
import re
from gslib.cs_api_map import ApiSelector
import gslib.tests.testcase as testcase
from gslib.tests.util import ObjectToURI as suri
from gslib.tests.util import unittest
from gslib.utils.retry_util import Retry
def _AssertEnabled(self, bucket_uri, value):
    stdout = self.RunGsUtil(self._get_ubla_cmd + [suri(bucket_uri)], return_stdout=True)
    uniform_bucket_level_access_re = re.compile('^\\s*Enabled:\\s+(?P<enabled_val>.+)$', re.MULTILINE)
    uniform_bucket_level_access_match = re.search(uniform_bucket_level_access_re, stdout)
    uniform_bucket_level_access_val = uniform_bucket_level_access_match.group('enabled_val')
    self.assertEqual(str(value), uniform_bucket_level_access_val)