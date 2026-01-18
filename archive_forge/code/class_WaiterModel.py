import logging
import time
import jmespath
from botocore.docs.docstring import WaiterDocstring
from botocore.utils import get_service_module_name
from . import xform_name
from .exceptions import ClientError, WaiterConfigError, WaiterError
class WaiterModel:
    SUPPORTED_VERSION = 2

    def __init__(self, waiter_config):
        """

        Note that the WaiterModel takes ownership of the waiter_config.
        It may or may not mutate the waiter_config.  If this is a concern,
        it is best to make a copy of the waiter config before passing it to
        the WaiterModel.

        :type waiter_config: dict
        :param waiter_config: The loaded waiter config
            from the <service>*.waiters.json file.  This can be
            obtained from a botocore Loader object as well.

        """
        self._waiter_config = waiter_config['waiters']
        version = waiter_config.get('version', 'unknown')
        self._verify_supported_version(version)
        self.version = version
        self.waiter_names = list(sorted(waiter_config['waiters'].keys()))

    def _verify_supported_version(self, version):
        if version != self.SUPPORTED_VERSION:
            raise WaiterConfigError(error_msg='Unsupported waiter version, supported version must be: %s, but version of waiter config is: %s' % (self.SUPPORTED_VERSION, version))

    def get_waiter(self, waiter_name):
        try:
            single_waiter_config = self._waiter_config[waiter_name]
        except KeyError:
            raise ValueError('Waiter does not exist: %s' % waiter_name)
        return SingleWaiterConfig(single_waiter_config)