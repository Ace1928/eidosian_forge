import base64
import datetime
import decimal
import inspect
import io
import logging
import netaddr
import re
import sys
import uuid
import weakref
from wsme import exc
class IPv6AddressType(UserType):
    """
    A simple IPv6 type.

    This type represents IPv6 addresses in the short format.
    """
    basetype = str
    name = 'ipv6address'

    @staticmethod
    def validate(value):
        try:
            netaddr.IPAddress(value, version=6, flags=netaddr.INET_PTON)
        except netaddr.AddrFormatError:
            error = 'Value should be IPv6 format'
            raise ValueError(error)
        else:
            return value