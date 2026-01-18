import os
import time
from twisted.internet import defer
from twisted.names import common, dns, error
from twisted.python import failure
from twisted.python.compat import execfile, nativeString
from twisted.python.filepath import FilePath
class FileAuthority(common.ResolverBase):
    """
    An Authority that is loaded from a file.

    This is an abstract class that implements record search logic. To create
    a functional resolver, subclass it and override the L{loadFile} method.

    @ivar _ADDITIONAL_PROCESSING_TYPES: Record types for which additional
        processing will be done.

    @ivar _ADDRESS_TYPES: Record types which are useful for inclusion in the
        additional section generated during additional processing.

    @ivar soa: A 2-tuple containing the SOA domain name as a L{bytes} and a
        L{dns.Record_SOA}.

    @ivar records: A mapping of domains (as lowercased L{bytes}) to records.
    @type records: L{dict} with L{bytes} keys
    """
    _ADDITIONAL_PROCESSING_TYPES = (dns.CNAME, dns.MX, dns.NS)
    _ADDRESS_TYPES = (dns.A, dns.AAAA)
    soa = None
    records = None

    def __init__(self, filename):
        common.ResolverBase.__init__(self)
        self.loadFile(filename)
        self._cache = {}

    def __setstate__(self, state):
        self.__dict__ = state

    def loadFile(self, filename):
        """
        Load DNS records from a file.

        This method populates the I{soa} and I{records} attributes. It must be
        overridden in a subclass. It is called once from the initializer.

        @param filename: The I{filename} parameter that was passed to the
        initilizer.

        @returns: L{None} -- the return value is ignored
        """

    def _additionalRecords(self, answer, authority, ttl):
        """
        Find locally known information that could be useful to the consumer of
        the response and construct appropriate records to include in the
        I{additional} section of that response.

        Essentially, implement RFC 1034 section 4.3.2 step 6.

        @param answer: A L{list} of the records which will be included in the
            I{answer} section of the response.

        @param authority: A L{list} of the records which will be included in
            the I{authority} section of the response.

        @param ttl: The default TTL for records for which this is not otherwise
            specified.

        @return: A generator of L{dns.RRHeader} instances for inclusion in the
            I{additional} section.  These instances represent extra information
            about the records in C{answer} and C{authority}.
        """
        for record in answer + authority:
            if record.type in self._ADDITIONAL_PROCESSING_TYPES:
                name = record.payload.name.name
                for rec in self.records.get(name.lower(), ()):
                    if rec.TYPE in self._ADDRESS_TYPES:
                        yield dns.RRHeader(name, rec.TYPE, dns.IN, rec.ttl or ttl, rec, auth=True)

    def _lookup(self, name, cls, type, timeout=None):
        """
        Determine a response to a particular DNS query.

        @param name: The name which is being queried and for which to lookup a
            response.
        @type name: L{bytes}

        @param cls: The class which is being queried.  Only I{IN} is
            implemented here and this value is presently disregarded.
        @type cls: L{int}

        @param type: The type of records being queried.  See the types defined
            in L{twisted.names.dns}.
        @type type: L{int}

        @param timeout: All processing is done locally and a result is
            available immediately, so the timeout value is ignored.

        @return: A L{Deferred} that fires with a L{tuple} of three sets of
            response records (to comprise the I{answer}, I{authority}, and
            I{additional} sections of a DNS response) or with a L{Failure} if
            there is a problem processing the query.
        """
        cnames = []
        results = []
        authority = []
        additional = []
        default_ttl = max(self.soa[1].minimum, self.soa[1].expire)
        domain_records = self.records.get(name.lower())
        if domain_records:
            for record in domain_records:
                if record.ttl is not None:
                    ttl = record.ttl
                else:
                    ttl = default_ttl
                if record.TYPE == dns.NS and name.lower() != self.soa[0].lower():
                    authority.append(dns.RRHeader(name, record.TYPE, dns.IN, ttl, record, auth=False))
                elif record.TYPE == type or type == dns.ALL_RECORDS:
                    results.append(dns.RRHeader(name, record.TYPE, dns.IN, ttl, record, auth=True))
                if record.TYPE == dns.CNAME:
                    cnames.append(dns.RRHeader(name, record.TYPE, dns.IN, ttl, record, auth=True))
            if not results:
                results = cnames
            additionalInformation = self._additionalRecords(results, authority, default_ttl)
            if cnames:
                results.extend(additionalInformation)
            else:
                additional.extend(additionalInformation)
            if not results and (not authority):
                authority.append(dns.RRHeader(self.soa[0], dns.SOA, dns.IN, ttl, self.soa[1], auth=True))
            return defer.succeed((results, authority, additional))
        elif dns._isSubdomainOf(name, self.soa[0]):
            return defer.fail(failure.Failure(dns.AuthoritativeDomainError(name)))
        else:
            return defer.fail(failure.Failure(error.DomainError(name)))

    def lookupZone(self, name, timeout=10):
        name = dns.domainString(name)
        if self.soa[0].lower() == name.lower():
            default_ttl = max(self.soa[1].minimum, self.soa[1].expire)
            if self.soa[1].ttl is not None:
                soa_ttl = self.soa[1].ttl
            else:
                soa_ttl = default_ttl
            results = [dns.RRHeader(self.soa[0], dns.SOA, dns.IN, soa_ttl, self.soa[1], auth=True)]
            for k, r in self.records.items():
                for rec in r:
                    if rec.ttl is not None:
                        ttl = rec.ttl
                    else:
                        ttl = default_ttl
                    if rec.TYPE != dns.SOA:
                        results.append(dns.RRHeader(k, rec.TYPE, dns.IN, ttl, rec, auth=True))
            results.append(results[0])
            return defer.succeed((results, (), ()))
        return defer.fail(failure.Failure(dns.DomainError(name)))

    def _cbAllRecords(self, results):
        ans, auth, add = ([], [], [])
        for res in results:
            if res[0]:
                ans.extend(res[1][0])
                auth.extend(res[1][1])
                add.extend(res[1][2])
        return (ans, auth, add)