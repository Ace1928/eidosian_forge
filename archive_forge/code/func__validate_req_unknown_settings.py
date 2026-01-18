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
def _validate_req_unknown_settings(self, **kwargs):
    """Checks if required settings are present.

        Also checks if unknown requirements are present.
        """
    self._all_attrs = self._req_settings | self._opt_settings
    if not kwargs and len(self._req_settings) > 0:
        raise MissingRequiredConf(desc='Missing all required attributes.')
    given_attrs = frozenset(kwargs.keys())
    unknown_attrs = given_attrs - self._all_attrs
    if unknown_attrs:
        raise RuntimeConfigError(desc='Unknown attributes: %s' % ', '.join([str(i) for i in unknown_attrs]))
    missing_req_settings = self._req_settings - given_attrs
    if missing_req_settings:
        raise MissingRequiredConf(conf_name=list(missing_req_settings))