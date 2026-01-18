from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import json
import os
from unittest import mock
from gslib.commands import web
import gslib.tests.testcase as testcase
from gslib.tests.testcase.integration_testcase import SkipForS3
from gslib.tests.util import ObjectToURI as suri
from gslib.tests.util import SetBotoConfigForTest
from gslib.tests.util import SetEnvironmentForTest
from gslib.utils import shim_util
class TestWebOldAlias(TestWeb):
    _set_web_cmd = ['setwebcfg']
    _get_web_cmd = ['getwebcfg']