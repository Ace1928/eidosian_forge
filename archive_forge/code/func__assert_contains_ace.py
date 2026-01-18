import os
import re
import tempfile
from os_win import _utils
from os_win import constants
from os_win.tests.functional import test_base
from os_win import utilsfactory
def _assert_contains_ace(self, path, access_to, access_flags):
    raw_out = self._get_raw_icacls_info(path)
    escaped_access_flags = access_flags.replace('(', '(?=.*\\(').replace(')', '\\))')
    pattern = '%s:%s.*' % (access_to, escaped_access_flags)
    match = re.findall(pattern, raw_out, flags=re.IGNORECASE | re.MULTILINE)
    if not match:
        fail_msg = 'The file does not contain the expected ACL rules. Raw icacls output: %s. Expected access rule: %s'
        expected_rule = ':'.join([access_to, access_flags])
        self.fail(fail_msg % (raw_out, expected_rule))