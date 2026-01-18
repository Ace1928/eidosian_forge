import email.utils
import os
import pickle
import time
from typing import Type
from twisted.application import internet
from twisted.internet import protocol
from twisted.internet.defer import Deferred, DeferredList
from twisted.internet.error import DNSLookupError
from twisted.internet.protocol import connectionDone
from twisted.mail import bounce, relay, smtp
from twisted.python import log
from twisted.python.failure import Failure
def _filterRecords(self, records):
    """
        Organize the records of a DNS response by record name.

        @type records: 3-L{tuple} of (0) L{list} of L{RRHeader
            <twisted.names.dns.RRHeader>}, (1) L{list} of L{RRHeader
            <twisted.names.dns.RRHeader>}, (2) L{list} of L{RRHeader
            <twisted.names.dns.RRHeader>}
        @param records: Answer resource records, authority resource records and
            additional resource records.

        @rtype: L{dict} mapping L{bytes} to L{list} of L{IRecord
            <twisted.names.dns.IRecord>} provider
        @return: A mapping of record name to record payload.
        """
    recordBag = {}
    for answer in records[0]:
        recordBag.setdefault(str(answer.name), []).append(answer.payload)
    return recordBag