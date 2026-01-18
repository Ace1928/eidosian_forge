import collections
import operator
import re
import warnings
import abc
from debtcollector import removals
import netaddr
import rfc3986
class HostAddress(ConfigType):
    """Host Address type.

    Represents both valid IP addresses and valid host domain names
    including fully qualified domain names.
    Performs strict checks for both IP addresses and valid hostnames,
    matching the opt values to the respective types as per RFC1912.

    :param version: defines which version should be explicitly
                    checked (4 or 6) in case of an IP address
    :param type_name: Type name to be used in the sample config file.
    """

    def __init__(self, version=None, type_name='host address value'):
        """Check for valid version in case an IP address is provided

        """
        super(HostAddress, self).__init__(type_name=type_name)
        self.ip_address = IPAddress(version, type_name)
        self.hostname = Hostname('localhost')

    def __call__(self, value):
        """Checks if is a valid IP/hostname.

        If not a valid IP, makes sure it is not a mistyped IP before
        performing checks for it as a hostname.

        """
        try:
            value = self.ip_address(value)
        except ValueError:
            try:
                value = self.hostname(value)
            except ValueError:
                raise ValueError('%s is not a valid host address' % (value,))
        return value

    def __repr__(self):
        return 'HostAddress'

    def __eq__(self, other):
        return self.__class__ == other.__class__

    def _formatter(self, value):
        return value