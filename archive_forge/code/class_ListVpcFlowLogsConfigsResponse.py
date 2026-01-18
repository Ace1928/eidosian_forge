from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ListVpcFlowLogsConfigsResponse(_messages.Message):
    """Response for the `ListVpcFlowLogsConfigs` method.

  Fields:
    nextPageToken: Page token to fetch the next set of configurations.
    unreachable: Locations that could not be reached (when querying all
      locations with `-`).
    vpcFlowLogsConfigs: List of VPC Flow Log configurations.
  """
    nextPageToken = _messages.StringField(1)
    unreachable = _messages.StringField(2, repeated=True)
    vpcFlowLogsConfigs = _messages.MessageField('VpcFlowLogsConfig', 3, repeated=True)