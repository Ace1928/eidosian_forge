import fnmatch
import getpass
import os
import re
import shlex
import socket
from hashlib import sha1
from io import StringIO
from functools import partial
from .ssh_exception import CouldNotCanonicalize, ConfigParseError
def _addressfamily_host_lookup(hostname, options):
    """
    Try looking up ``hostname`` in an IPv4 or IPv6 specific manner.

    This is an odd duck due to needing use in two divergent use cases. It looks
    up ``AddressFamily`` in ``options`` and if it is ``inet`` or ``inet6``,
    this function uses `socket.getaddrinfo` to perform a family-specific
    lookup, returning the result if successful.

    In any other situation -- lookup failure, or ``AddressFamily`` being
    unspecified or ``any`` -- ``None`` is returned instead and the caller is
    expected to do something situation-appropriate like calling
    `socket.gethostbyname`.

    :param str hostname: Hostname to look up.
    :param options: `SSHConfigDict` instance w/ parsed options.
    :returns: ``getaddrinfo``-style tuples, or ``None``, depending.
    """
    address_family = options.get('addressfamily', 'any').lower()
    if address_family == 'any':
        return
    try:
        family = socket.AF_INET6
        if address_family == 'inet':
            family = socket.AF_INET
        return socket.getaddrinfo(hostname, None, family, socket.SOCK_DGRAM, socket.IPPROTO_IP, socket.AI_CANONNAME)
    except socket.gaierror:
        pass