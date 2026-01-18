import collections
import operator
import re
import warnings
import abc
from debtcollector import removals
import netaddr
import rfc3986
def _check_both_versions(self, address):
    if not (netaddr.valid_ipv4(address, netaddr.core.INET_PTON) or netaddr.valid_ipv6(address, netaddr.core.INET_PTON)):
        raise ValueError('%s is not IPv4 or IPv6 address' % address)