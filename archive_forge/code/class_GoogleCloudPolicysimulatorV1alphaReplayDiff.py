from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudPolicysimulatorV1alphaReplayDiff(_messages.Message):
    """The difference between the results of evaluating an access tuple under
  the current (baseline) policies and under the proposed (simulated) policies.
  This difference explains how a principal's access could change if the
  proposed policies were applied.

  Fields:
    accessDiff: A summary and comparison of the principal's access under the
      current (baseline) policies and the proposed (simulated) policies for a
      single access tuple. The evaluation of the principal's access is
      reported in the AccessState field.
  """
    accessDiff = _messages.MessageField('GoogleCloudPolicysimulatorV1alphaAccessStateDiff', 1)