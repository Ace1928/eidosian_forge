import abc
import functools
import os
import re
from pyudev._errors import DeviceNotFoundError
from pyudev.device import Devices
class DevicePathHypothesis(Hypothesis):
    """
    Discover the device assuming the identifier is a device path.
    """

    @classmethod
    def match(cls, value):
        """
        Match ``value`` under the assumption that it is a device path.

        :returns: the device path or None
        :rtype: str or NoneType
        """
        return value

    @classmethod
    def lookup(cls, context, key):
        """
        Lookup by the path.

        :param Context context: the context
        :param str key: the device path
        :returns: a list of matching devices
        :rtype: frozenset of :class:`Device`
        """
        res = wrap_exception(Devices.from_path)(context, key)
        return frozenset((res,)) if res is not None else frozenset()