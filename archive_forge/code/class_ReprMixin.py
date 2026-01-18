import os
import binascii
from typing import List
from libcloud.utils.retry import DEFAULT_DELAY  # noqa: F401
from libcloud.utils.retry import DEFAULT_BACKOFF  # noqa: F401
from libcloud.utils.retry import DEFAULT_TIMEOUT  # noqa: F401
from libcloud.utils.retry import TRANSIENT_SSL_ERROR  # noqa: F401
from libcloud.utils.retry import Retry  # flake8: noqa
from libcloud.utils.retry import TransientSSLError  # noqa: F401
from libcloud.common.providers import get_driver as _get_driver
from libcloud.common.providers import set_driver as _set_driver
class ReprMixin:
    """
    Mixin class which adds __repr__ and __str__ methods for the attributes
    specified on the class.
    """
    _repr_attributes = []

    def __repr__(self):
        attributes = []
        for attribute in self._repr_attributes:
            value = getattr(self, attribute, None)
            attributes.append('{}={}'.format(attribute, value))
        values = (self.__class__.__name__, ', '.join(attributes))
        result = '<%s %s>' % values
        return result

    def __str__(self):
        return str(self.__repr__())