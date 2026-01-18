from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.core import exceptions
class DeviceConfigsClient(object):
    """Client for device_configs service in the Cloud IoT API."""

    def __init__(self, client=None, messages=None):
        self.client = client or GetClientInstance()
        self.messages = messages or GetMessagesModule(client)
        service = self.client.projects_locations_registries_devices_configVersions
        self._service = service

    def List(self, parent_ref, num_versions=None):
        """List all device configurations available for a device.

    Up to a maximum of 10 (enforced by service). No pagination.

    Args:
      parent_ref: a Resource reference to a
        cloudiot.projects.locations.registries.devices resource.
      num_versions: int, the number of device configurations to list (max 10).

    Returns:
      List of DeviceConfig
    """
        request_type = getattr(self.messages, 'CloudiotProjectsLocationsRegistriesDevicesConfigVersionsListRequest')
        response = self._service.List(request_type(name=parent_ref.RelativeName(), numVersions=num_versions))
        return response.deviceConfigs