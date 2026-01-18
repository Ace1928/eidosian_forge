import collections
import operator
import re
import warnings
import abc
from debtcollector import removals
import netaddr
import rfc3986
class HostDomain(HostAddress):
    """Host Domain type.

    Like HostAddress with the support of _ character.

    :param version: defines which version should be explicitly
                    checked (4 or 6) in case of an IP address
    :param type_name: Type name to be used in the sample config file.
    """
    DOMAIN_REGEX = '(?!-)[A-Z0-9-_]{1,63}(?<!-)$'

    def __init__(self, version=None, type_name='host domain value'):
        """Check for valid version in case an IP address is provided

        """
        super(HostDomain, self).__init__(version=version, type_name=type_name)

    def __call__(self, value):
        """Checks if is a valid IP/hostname.

        If not a valid IP, makes sure it is not a mistyped IP before
        performing checks for it as a hostname.

        """
        try:
            value = super(HostDomain, self).__call__(value)
        except ValueError:
            try:
                value = self.hostname(value, regex=self.DOMAIN_REGEX)
            except ValueError:
                raise ValueError('%s is not a valid host address' % (value,))
        return value

    def __repr__(self):
        return 'HostDomain'