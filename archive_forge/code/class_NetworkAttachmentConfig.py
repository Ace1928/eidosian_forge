from .. import errors
from ..constants import IS_WINDOWS_PLATFORM
from ..utils import (
class NetworkAttachmentConfig(dict):
    """
        Network attachment options for a service.

        Args:
            target (str): The target network for attachment.
                Can be a network name or ID.
            aliases (:py:class:`list`): A list of discoverable alternate names
                for the service.
            options (:py:class:`dict`): Driver attachment options for the
                network target.
    """

    def __init__(self, target, aliases=None, options=None):
        self['Target'] = target
        self['Aliases'] = aliases
        self['DriverOpts'] = options