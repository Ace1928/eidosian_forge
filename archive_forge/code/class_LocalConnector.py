from __future__ import annotations
from os_brick.i18n import _
from os_brick.initiator.connectors import base
from os_brick import utils
class LocalConnector(base.BaseLinuxConnector):
    """"Connector class to attach/detach File System backed volumes."""

    def __init__(self, root_helper, driver=None, *args, **kwargs):
        super(LocalConnector, self).__init__(root_helper, *args, driver=driver, **kwargs)

    @staticmethod
    def get_connector_properties(root_helper: str, *args, **kwargs) -> dict:
        """The Local connector properties."""
        return {}

    def get_volume_paths(self, connection_properties: dict) -> list[str]:
        path = connection_properties['device_path']
        return [path]

    def get_search_path(self):
        return None

    def get_all_available_volumes(self, connection_properties=None):
        return []

    @utils.trace
    def connect_volume(self, connection_properties: dict) -> dict:
        """Connect to a volume.

        :param connection_properties: The dictionary that describes all of the
          target volume attributes. ``connection_properties`` must include:

          - ``device_path`` - path to the volume to be connected
        :type connection_properties: dict
        :returns: dict
        """
        if 'device_path' not in connection_properties:
            msg = _('Invalid connection_properties specified no device_path attribute')
            raise ValueError(msg)
        device_info = {'type': 'local', 'path': connection_properties['device_path']}
        return device_info

    @utils.trace
    def disconnect_volume(self, connection_properties, device_info, force=False, ignore_errors=False):
        """Disconnect a volume from the local host.

        :param connection_properties: The dictionary that describes all
                                      of the target volume attributes.
        :type connection_properties: dict
        :param device_info: historical difference, but same as connection_props
        :type device_info: dict
        """
        pass

    def extend_volume(self, connection_properties):
        raise NotImplementedError