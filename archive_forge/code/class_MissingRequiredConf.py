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
@add_bgp_error_metadata(code=RUNTIME_CONF_ERROR_CODE, sub_code=2, def_desc='Missing required configuration.')
class MissingRequiredConf(RuntimeConfigError):
    """Exception raised when trying to configure with missing required
    settings.
    """

    def __init__(self, **kwargs):
        conf_name = kwargs.get('conf_name')
        if conf_name:
            super(MissingRequiredConf, self).__init__(desc='Missing required configuration: %s' % conf_name)
        else:
            super(MissingRequiredConf, self).__init__(desc=kwargs.get('desc'))