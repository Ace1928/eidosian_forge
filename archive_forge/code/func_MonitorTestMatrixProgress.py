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
def MonitorTestMatrixProgress(self):
    """Monitor and report the progress of multiple running tests in a matrix."""
    while True:
        matrix = self.GetTestMatrixStatus()
        state_counts = collections.defaultdict(int)
        for test in matrix.testExecutions:
            state_counts[test.state] += 1
        self._UpdateMatrixStatus(state_counts)
        if matrix.state in self.completed_matrix_states:
            self._LogTestComplete(matrix.state)
            break
        self._SleepForStatusInterval()