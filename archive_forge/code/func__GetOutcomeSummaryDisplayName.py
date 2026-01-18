from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
from apitools.base.py import exceptions as apitools_exceptions
from googlecloudsdk.api_lib.firebase.test import util
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.core import log
def _GetOutcomeSummaryDisplayName(self, outcome):
    """Transforms the outcome enum to a human readable outcome.

    Args:
      outcome: An Outcome.SummaryValueValuesEnum value.

    Returns:
      A string containing a human readable outcome.
    """
    try:
        return self._outcome_names[outcome]
    except ValueError:
        return 'Unknown'