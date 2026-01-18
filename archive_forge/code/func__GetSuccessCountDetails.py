from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
from apitools.base.py import exceptions as apitools_exceptions
from googlecloudsdk.api_lib.firebase.test import util
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.core import log
def _GetSuccessCountDetails(test_suite_overviews):
    """Build a string with status count sums for testSuiteOverviews."""
    total = 0
    skipped = 0
    for overview in test_suite_overviews:
        total += overview.totalCount or 0
        skipped += overview.skippedCount or 0
    passed = total - skipped
    if passed:
        msg = '{p} test cases passed'.format(p=passed)
        if skipped:
            msg = '{m}, {s} skipped'.format(m=msg, s=skipped)
        return msg
    return '--'