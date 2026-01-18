from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudNetworkconnectivityV1betaSpokeSummary(_messages.Message):
    """Summarizes information about the spokes associated with a hub. The
  summary includes a count of spokes according to type and according to state.
  If any spokes are inactive, the summary also lists the reasons they are
  inactive, including a count for each reason.

  Fields:
    spokeStateCounts: Output only. Counts the number of spokes that are in
      each state and associated with a given hub.
    spokeStateReasonCounts: Output only. Counts the number of spokes that are
      inactive for each possible reason and associated with a given hub.
    spokeTypeCounts: Output only. Counts the number of spokes of each type
      that are associated with a specific hub.
  """
    spokeStateCounts = _messages.MessageField('GoogleCloudNetworkconnectivityV1betaSpokeStateCount', 1, repeated=True)
    spokeStateReasonCounts = _messages.MessageField('GoogleCloudNetworkconnectivityV1betaSpokeStateReasonCount', 2, repeated=True)
    spokeTypeCounts = _messages.MessageField('GoogleCloudNetworkconnectivityV1betaSpokeTypeCount', 3, repeated=True)