from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.core import exceptions
class DeviceStatesClient(object):
    """Client for device_states service in the Cloud IoT API."""

    def __init__(self, client=None, messages=None):
        self.client = client or GetClientInstance()
        self.messages = messages or GetMessagesModule(client)
        self._service = self.client.projects_locations_registries_devices_states

    def List(self, parent_ref, num_states=None):
        """List all device states available for a device.

    Up to a maximum of 10 (enforced by service). No pagination.

    Args:
      parent_ref: a Resource reference to a
        cloudiot.projects.locations.registries.devices resource.
      num_states: int, the number of device states to list (max 10).

    Returns:
      List of DeviceStates
    """
        request_type = getattr(self.messages, 'CloudiotProjectsLocationsRegistriesDevicesStatesListRequest')
        response = self._service.List(request_type(name=parent_ref.RelativeName(), numStates=num_states))
        return response.deviceStates