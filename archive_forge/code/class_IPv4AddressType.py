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
class IPv4AddressType(UserType):
    """
    A simple IPv4 type.
    """
    basetype = str
    name = 'ipv4address'

    @staticmethod
    def validate(value):
        try:
            netaddr.IPAddress(value, version=4, flags=netaddr.INET_PTON)
        except netaddr.AddrFormatError:
            error = 'Value should be IPv4 format'
            raise ValueError(error)
        else:
            return value