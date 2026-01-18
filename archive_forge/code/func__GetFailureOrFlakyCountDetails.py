from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
from apitools.base.py import exceptions as apitools_exceptions
from googlecloudsdk.api_lib.firebase.test import util
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.core import log
def _GetFailureOrFlakyCountDetails(test_suite_overviews):
    """Build a string with status count sums for testSuiteOverviews."""
    total = 0
    error = 0
    failed = 0
    skipped = 0
    flaky = 0
    for overview in test_suite_overviews:
        total += overview.totalCount or 0
        error += overview.errorCount or 0
        failed += overview.failureCount or 0
        skipped += overview.skippedCount or 0
        flaky += overview.flakyCount or 0
    if total:
        msg = '{f} test cases failed'.format(f=failed)
        passed = total - error - failed - skipped - flaky
        if flaky and failed:
            msg = '{m}, {f} flaky'.format(m=msg, f=flaky)
        if flaky and (not failed):
            msg = '{f} test cases flaky'.format(f=flaky)
        if passed:
            msg = '{m}, {p} passed'.format(m=msg, p=passed)
        if error:
            msg = '{m}, {e} errors'.format(m=msg, e=error)
        if skipped:
            msg = '{m}, {s} skipped'.format(m=msg, s=skipped)
        return msg
    else:
        return 'Test failed to run'