from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class NetworkEndpointGroupsListEndpointsRequest(_messages.Message):
    """A NetworkEndpointGroupsListEndpointsRequest object.

  Enums:
    HealthStatusValueValuesEnum: Optional query parameter for showing the
      health status of each network endpoint. Valid options are SKIP or SHOW.
      If you don't specify this parameter, the health status of network
      endpoints will not be provided.

  Fields:
    endpointFilters: Optional list of endpoints to query. This is a more
      efficient but also limited version of filter parameter. Endpoints in the
      filter must have ip_address and port fields populated, other fields are
      not supported.
    healthStatus: Optional query parameter for showing the health status of
      each network endpoint. Valid options are SKIP or SHOW. If you don't
      specify this parameter, the health status of network endpoints will not
      be provided.
  """

    class HealthStatusValueValuesEnum(_messages.Enum):
        """Optional query parameter for showing the health status of each network
    endpoint. Valid options are SKIP or SHOW. If you don't specify this
    parameter, the health status of network endpoints will not be provided.

    Values:
      SHOW: Show the health status for each network endpoint. Impacts latency
        of the call.
      SKIP: Health status for network endpoints will not be provided.
    """
        SHOW = 0
        SKIP = 1
    endpointFilters = _messages.MessageField('NetworkEndpointGroupsListEndpointsRequestNetworkEndpointFilter', 1, repeated=True)
    healthStatus = _messages.EnumField('HealthStatusValueValuesEnum', 2)