from __future__ import (absolute_import, division, print_function)
import re
from ansible.errors import AnsibleParserError, AnsibleError

    Takes a string and returns a (host, port) tuple. If the host is None, then
    the string could not be parsed as a host identifier with an optional port
    specification. If the port is None, then no port was specified.

    The host identifier may be a hostname (qualified or not), an IPv4 address,
    or an IPv6 address. If allow_ranges is True, then any of those may contain
    [x:y] range specifications, e.g. foo[1:3] or foo[0:5]-bar[x-z].

    The port number is an optional :NN suffix on an IPv4 address or host name,
    or a mandatory :NN suffix on any square-bracketed expression: IPv6 address,
    IPv4 address, or host name. (This means the only way to specify a port for
    an IPv6 address is to enclose it in square brackets.)
    