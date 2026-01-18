from os_brick.i18n import _
from os_brick.initiator.connectors import local
from os_brick import utils
class GPFSConnector(local.LocalConnector):
    """"Connector class to attach/detach File System backed volumes."""

    @utils.trace
    def connect_volume(self, connection_properties):
        """Connect to a volume.

        :param connection_properties: The dictionary that describes all
                                      of the target volume attributes.
               connection_properties must include:
               device_path - path to the volume to be connected
        :type connection_properties: dict
        :returns: dict
        """
        if 'device_path' not in connection_properties:
            msg = _('Invalid connection_properties specified no device_path attribute.')
            raise ValueError(msg)
        device_info = {'type': 'gpfs', 'path': connection_properties['device_path']}
        return device_info