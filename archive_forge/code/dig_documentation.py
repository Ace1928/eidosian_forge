from __future__ import (absolute_import, division, print_function)
from ansible.errors import AnsibleError
from ansible.plugins.lookup import LookupBase
from ansible.module_utils.common.text.converters import to_native
from ansible.module_utils.parsing.convert_bool import boolean
from ansible.utils.display import Display
import socket

        terms contains a string with things to `dig' for. We support the
        following formats:
            example.com                                     # A record
            example.com  qtype=A                            # same
            example.com/TXT                                 # specific qtype
            example.com  qtype=txt                          # same
            192.0.2.23/PTR                                 # reverse PTR
              ^^ shortcut for 23.2.0.192.in-addr.arpa/PTR
            example.net/AAAA  @nameserver                   # query specified server
                               ^^^ can be comma-sep list of names/addresses

            ... flat=0                                      # returns a dict; default is 1 == string
        