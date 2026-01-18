import socket
import sys
import time
import random
import dns.exception
import dns.flags
import dns.ipv4
import dns.ipv6
import dns.message
import dns.name
import dns.query
import dns.rcode
import dns.rdataclass
import dns.rdatatype
import dns.reversename
import dns.tsig
from ._compat import xrange, string_types
def _config_win32_nameservers(self, nameservers):
    nameservers = str(nameservers)
    split_char = self._determine_split_char(nameservers)
    ns_list = nameservers.split(split_char)
    for ns in ns_list:
        if ns not in self.nameservers:
            self.nameservers.append(ns)