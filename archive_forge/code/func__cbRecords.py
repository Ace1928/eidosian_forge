import socket
from zope.interface import implementer
from twisted.internet import defer, error, interfaces
from twisted.logger import Logger
from twisted.names import dns
from twisted.names.error import (
def _cbRecords(self, records, name, effort):
    ans, auth, add = records
    result = extractRecord(self, dns.Name(name), ans + auth + add, effort)
    if not result:
        raise error.DNSLookupError(name)
    return result