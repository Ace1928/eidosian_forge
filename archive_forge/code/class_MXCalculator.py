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
class MXCalculator:
    """
    A utility for looking up mail exchange hosts and tracking whether they are
    working or not.

    @type clock: L{IReactorTime <twisted.internet.interfaces.IReactorTime>}
        provider
    @ivar clock: A reactor which will be used to schedule timeouts.

    @type resolver: L{IResolver <twisted.internet.interfaces.IResolver>}
    @ivar resolver: A resolver.

    @type badMXs: L{dict} mapping L{bytes} to L{float}
    @ivar badMXs: A mapping of non-functioning mail exchange hostname to time
        at which another attempt at contacting it may be made.

    @type timeOutBadMX: L{int}
    @ivar timeOutBadMX: Period in seconds between attempts to contact a
        non-functioning mail exchange host.

    @type fallbackToDomain: L{bool}
    @ivar fallbackToDomain: A flag indicating whether to attempt to use the
        hostname directly when no mail exchange can be found (C{True}) or
        not (C{False}).
    """
    timeOutBadMX = 60 * 60
    fallbackToDomain = True

    def __init__(self, resolver=None, clock=None):
        """
        @type resolver: L{IResolver <twisted.internet.interfaces.IResolver>}
            provider or L{None}
        @param resolver: A resolver.

        @type clock: L{IReactorTime <twisted.internet.interfaces.IReactorTime>}
            provider or L{None}
        @param clock: A reactor which will be used to schedule timeouts.
        """
        self.badMXs = {}
        if resolver is None:
            from twisted.names.client import createResolver
            resolver = createResolver()
        self.resolver = resolver
        if clock is None:
            from twisted.internet import reactor as clock
        self.clock = clock

    def markBad(self, mx):
        """
        Record that a mail exchange host is not currently functioning.

        @type mx: L{bytes}
        @param mx: The hostname of a mail exchange host.
        """
        self.badMXs[str(mx)] = self.clock.seconds() + self.timeOutBadMX

    def markGood(self, mx):
        """
        Record that a mail exchange host is functioning.

        @type mx: L{bytes}
        @param mx: The hostname of a mail exchange host.
        """
        try:
            del self.badMXs[mx]
        except KeyError:
            pass

    def getMX(self, domain, maximumCanonicalChainLength=3):
        """
        Find the name of a host that acts as a mail exchange server
        for a domain.

        @type domain: L{bytes}
        @param domain: A domain name.

        @type maximumCanonicalChainLength: L{int}
        @param maximumCanonicalChainLength: The maximum number of unique
            canonical name records to follow while looking up the mail exchange
            host.

        @rtype: L{Deferred} which successfully fires with L{Record_MX}
        @return: A deferred which succeeds with the MX record for the mail
            exchange server for the domain or fails if none can be found.
        """
        mailExchangeDeferred = self.resolver.lookupMailExchange(domain)
        mailExchangeDeferred.addCallback(self._filterRecords)
        mailExchangeDeferred.addCallback(self._cbMX, domain, maximumCanonicalChainLength)
        mailExchangeDeferred.addErrback(self._ebMX, domain)
        return mailExchangeDeferred

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

    def _ebMX(self, failure, domain):
        """
        Attempt to use the name of the domain directly when mail exchange
        lookup fails.

        @type failure: L{Failure}
        @param failure: The reason for the lookup failure.

        @type domain: L{bytes}
        @param domain: The domain name.

        @rtype: L{Record_MX <twisted.names.dns.Record_MX>} or L{Failure}
        @return: An MX record for the domain or a failure if the fallback to
            domain option is not in effect and an error, other than not
            finding an MX record, occurred during lookup.

        @raise IOError: When no MX record could be found and the fallback to
            domain option is not in effect.

        @raise DNSLookupError: When no MX record could be found and the
            fallback to domain option is in effect but no address for the
            domain could be found.
        """
        from twisted.names import dns, error
        if self.fallbackToDomain:
            failure.trap(error.DNSNameError)
            log.msg('MX lookup failed; attempting to use hostname ({}) directly'.format(domain))
            d = self.resolver.getHostByName(domain)

            def cbResolved(addr):
                return dns.Record_MX(name=addr)

            def ebResolved(err):
                err.trap(error.DNSNameError)
                raise DNSLookupError()
            d.addCallbacks(cbResolved, ebResolved)
            return d
        elif failure.check(error.DNSNameError):
            raise OSError(f'No MX found for {domain!r}')
        return failure