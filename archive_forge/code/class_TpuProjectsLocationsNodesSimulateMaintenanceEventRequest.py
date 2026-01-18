from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class TpuProjectsLocationsNodesSimulateMaintenanceEventRequest(_messages.Message):
    """A TpuProjectsLocationsNodesSimulateMaintenanceEventRequest object.

  Fields:
    name: Required. The resource name.
    simulateMaintenanceEventRequest: A SimulateMaintenanceEventRequest
      resource to be passed as the request body.
  """
    name = _messages.StringField(1, required=True)
    simulateMaintenanceEventRequest = _messages.MessageField('SimulateMaintenanceEventRequest', 2)