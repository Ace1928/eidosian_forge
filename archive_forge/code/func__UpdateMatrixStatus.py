from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import datetime
import time
from apitools.base.py import exceptions as apitools_exceptions
from googlecloudsdk.api_lib.firebase.test import exceptions
from googlecloudsdk.api_lib.firebase.test import util
from googlecloudsdk.calliope import exceptions as calliope_exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.console import console_io
import six
def _UpdateMatrixStatus(self, state_counts):
    """Update the matrix status line with the current test state counts.

    Example: 'Test matrix status: Finished:5 Running:3 Unsupported:2'

    Args:
      state_counts: {state:count} a dict mapping a test state to its frequency.
    """
    status = []
    timestamp = self._clock().strftime(_TIMESTAMP_FORMAT)
    for state, count in six.iteritems(state_counts):
        if count > 0:
            status.append('{s}:{c}'.format(s=self._state_names[state], c=count))
    status.sort()
    out = '\r{0} Test matrix status: {1} '.format(timestamp, ' '.join(status))
    self._max_status_length = max(len(out), self._max_status_length)
    log.status.write(out.ljust(self._max_status_length))