from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import os
import re
import sys
from unittest import mock
from gslib.exception import NO_URLS_MATCHED_PREFIX
from gslib.exception import NO_URLS_MATCHED_TARGET
import gslib.tests.testcase as testcase
from gslib.tests.testcase.base import MAX_BUCKET_LENGTH
from gslib.tests.testcase.integration_testcase import SkipForS3
import gslib.tests.util as util
from gslib.tests.util import GenerationFromURI as urigen
from gslib.tests.util import ObjectToURI as suri
from gslib.tests.util import SetBotoConfigForTest
from gslib.tests.util import SetEnvironmentForTest
from gslib.utils import shim_util
from gslib.utils.retry_util import Retry
def _CleanRmUiOutputBeforeChecking(self, stderr):
    """Excludes everything coming from the UI to avoid assert errors.

    Args:
      stderr: The cumulative stderr output.
    Returns:
      The cumulative stderr output without the expected UI output.
    """
    if self._use_gcloud_storage:
        return self._CleanOutputLinesForGcloudStorage(stderr)
    ui_output_pattern = '[^\n\r]*objects][^\n\r]*[\n\r]'
    final_message_pattern = 'Operation completed over[^\n]*'
    ui_spinner_list = ['\\\r', '|\r', '/\r', '-\r']
    ui_lines_list = re.findall(ui_output_pattern, stderr) + re.findall(final_message_pattern, stderr) + ui_spinner_list
    for ui_line in ui_lines_list:
        stderr = stderr.replace(ui_line, '')
    return stderr