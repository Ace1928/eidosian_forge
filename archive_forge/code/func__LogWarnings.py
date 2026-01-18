from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
from apitools.base.py import exceptions as apitools_exceptions
from googlecloudsdk.api_lib.firebase.test import util
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.core import log
def _LogWarnings(self, details, axis_value):
    """Log warnings if there was native crash or infrustructure failure."""
    if _NATIVE_CRASH in details:
        log.warning(_NATIVE_CRASH_DETAILED_FORMAT.format(axis_value))
    if _INFRASTRUCTURE_FAILURE in details:
        log.warning(_INFRASTRUCTURE_FAILURE_DETAILED_FORMAT.format(axis_value, self._test_matrix_id))