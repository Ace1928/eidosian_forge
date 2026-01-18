from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
from apitools.base.py import exceptions as apitools_exceptions
from googlecloudsdk.api_lib.firebase.test import util
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.core import log
def _GetInconclusiveDetail(outcome):
    """Build a string with inconclusiveDetail if present."""
    if outcome.inconclusiveDetail:
        if outcome.inconclusiveDetail.infrastructureFailure:
            return _INFRASTRUCTURE_FAILURE
        if outcome.inconclusiveDetail.abortedByUser:
            return 'Test run aborted by user'
    return 'Unknown reason'