from .. import errors
from ..constants import IS_WINDOWS_PLATFORM
from ..utils import (
class DriverConfig(dict):
    """
    Indicates which driver to use, as well as its configuration. Can be used
    as ``log_driver`` in a :py:class:`~docker.types.ContainerSpec`,
    for the `driver_config` in a volume :py:class:`~docker.types.Mount`, or
    as the driver object in
    :py:meth:`create_secret`.

    Args:

        name (string): Name of the driver to use.
        options (dict): Driver-specific options. Default: ``None``.
    """

    def __init__(self, name, options=None):
        self['Name'] = name
        if options:
            self['Options'] = options