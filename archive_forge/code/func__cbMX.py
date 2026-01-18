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
def _cbMX(self, answers, domain, cnamesLeft):
    """
        Try to find the mail exchange host for a domain from the given DNS
        records.

        This will attempt to resolve canonical name record results.  It can
        recognize loops and will give up on non-cyclic chains after a specified
        number of lookups.

        @type answers: L{dict} mapping L{bytes} to L{list} of L{IRecord
            <twisted.names.dns.IRecord>} provider
        @param answers: A mapping of record name to record payload.

        @type domain: L{bytes}
        @param domain: A domain name.

        @type cnamesLeft: L{int}
        @param cnamesLeft: The number of unique canonical name records
            left to follow while looking up the mail exchange host.

        @rtype: L{Record_MX <twisted.names.dns.Record_MX>} or L{Failure}
        @return: An MX record for the mail exchange host or a failure if one
            cannot be found.
        """
    from twisted.names import dns, error
    seenAliases = set()
    exchanges = []
    pertinentRecords = answers.get(domain, [])
    while pertinentRecords:
        record = pertinentRecords.pop()
        if record.TYPE == dns.CNAME:
            seenAliases.add(domain)
            canonicalName = str(record.name)
            if canonicalName in answers:
                if canonicalName in seenAliases:
                    return Failure(CanonicalNameLoop(record))
                pertinentRecords = answers[canonicalName]
                exchanges = []
            elif cnamesLeft:
                return self.getMX(canonicalName, cnamesLeft - 1)
            else:
                return Failure(CanonicalNameChainTooLong(record))
        if record.TYPE == dns.MX:
            exchanges.append((record.preference, record))
    if exchanges:
        exchanges.sort()
        for preference, record in exchanges:
            host = str(record.name)
            if host not in self.badMXs:
                return record
            t = self.clock.seconds() - self.badMXs[host]
            if t >= 0:
                del self.badMXs[host]
                return record
        return exchanges[0][1]
    else:
        return Failure(error.DNSNameError(f'No MX records for {domain!r}'))