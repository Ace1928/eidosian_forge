from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
from apitools.base.py import exceptions as apitools_exceptions
from googlecloudsdk.api_lib.firebase.test import util
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.core import log
def _TestOutcomeSortKey(x):
    """Transform a TestOutcome to a tuple yielding the desired sort order."""
    return tuple([_OUTCOME_SORTING[x.outcome], x.test_details, x.axis_value])