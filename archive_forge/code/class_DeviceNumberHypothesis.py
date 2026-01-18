import abc
import functools
import os
import re
from pyudev._errors import DeviceNotFoundError
from pyudev.device import Devices
class DeviceNumberHypothesis(Hypothesis):
    """
    Represents the hypothesis that the device is a device number.

    The device may be separated into major/minor number or a composite number.
    """

    @classmethod
    def _match_major_minor(cls, value):
        """
        Match the number under the assumption that it is a major,minor pair.

        :param str value: value to match
        :returns: the device number or None
        :rtype: int or NoneType
        """
        major_minor_re = re.compile('^(?P<major>\\d+)(\\D+)(?P<minor>\\d+)$')
        match = major_minor_re.match(value)
        return match and os.makedev(int(match.group('major')), int(match.group('minor')))

    @classmethod
    def _match_number(cls, value):
        """
        Match the number under the assumption that it is a single number.

        :param str value: value to match
        :returns: the device number or None
        :rtype: int or NoneType
        """
        number_re = re.compile('^(?P<number>\\d+)$')
        match = number_re.match(value)
        return match and int(match.group('number'))

    @classmethod
    def match(cls, value):
        """
        Match the number under the assumption that it is a device number.

        :returns: the device number or None
        :rtype: int or NoneType
        """
        return cls._match_major_minor(value) or cls._match_number(value)

    @classmethod
    def find_subsystems(cls, context):
        """
        Find subsystems in /sys/dev.

        :param Context context: the context
        :returns: a lis of available subsystems
        :rtype: list of str
        """
        sys_path = context.sys_path
        return os.listdir(os.path.join(sys_path, 'dev'))

    @classmethod
    def lookup(cls, context, key):
        """
        Lookup by the device number.

        :param Context context: the context
        :param int key: the device number
        :returns: a list of matching devices
        :rtype: frozenset of :class:`Device`
        """
        func = wrap_exception(Devices.from_device_number)
        res = (func(context, s, key) for s in cls.find_subsystems(context))
        return frozenset((r for r in res if r is not None))