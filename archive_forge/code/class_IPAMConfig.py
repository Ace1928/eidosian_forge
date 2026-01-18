from .. import errors
from ..utils import normalize_links, version_lt
class IPAMConfig(dict):
    """
    Create an IPAM (IP Address Management) config dictionary to be used with
    :py:meth:`~docker.api.network.NetworkApiMixin.create_network`.

    Args:

        driver (str): The IPAM driver to use. Defaults to ``default``.
        pool_configs (:py:class:`list`): A list of pool configurations
          (:py:class:`~docker.types.IPAMPool`). Defaults to empty list.
        options (dict): Driver options as a key-value dictionary.
          Defaults to `None`.

    Example:

        >>> ipam_config = docker.types.IPAMConfig(driver='default')
        >>> network = client.create_network('network1', ipam=ipam_config)

    """

    def __init__(self, driver='default', pool_configs=None, options=None):
        self.update({'Driver': driver, 'Config': pool_configs or []})
        if options:
            if not isinstance(options, dict):
                raise TypeError('IPAMConfig options must be a dictionary')
            self['Options'] = options