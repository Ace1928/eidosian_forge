import abc
from abc import ABCMeta
from abc import abstractmethod
import functools
import numbers
import logging
import uuid
from os_ken.services.protocols.bgp.base import add_bgp_error_metadata
from os_ken.services.protocols.bgp.base import BGPSException
from os_ken.services.protocols.bgp.base import get_validator
from os_ken.services.protocols.bgp.base import RUNTIME_CONF_ERROR_CODE
from os_ken.services.protocols.bgp.base import validate
from os_ken.services.protocols.bgp.utils import validation
from os_ken.services.protocols.bgp.utils.validation import is_valid_asn
@add_bgp_error_metadata(code=RUNTIME_CONF_ERROR_CODE, sub_code=4, def_desc='Incorrect Value for configuration.')
class ConfigValueError(RuntimeConfigError):
    """Exception raised when configuration value is of correct type but
    incorrect value.
    """

    def __init__(self, **kwargs):
        conf_name = kwargs.get(CONF_NAME)
        conf_value = kwargs.get(CONF_VALUE)
        if conf_name and conf_value:
            super(ConfigValueError, self).__init__(desc='Incorrect Value %s for configuration: %s' % (conf_value, conf_name))
        elif conf_name:
            super(ConfigValueError, self).__init__(desc='Incorrect Value for configuration: %s' % conf_name)
        else:
            super(ConfigValueError, self).__init__(desc=kwargs.get('desc'))