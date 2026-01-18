from twisted.application import service
from twisted.internet import defer, task
from twisted.names import client, common, dns, resolve
from twisted.names.authority import FileAuthority
from twisted.python import failure, log
from twisted.python.compat import nativeString
class SecondaryAuthority(FileAuthority):
    """
    An Authority that keeps itself updated by performing zone transfers.

    @ivar primary: The IP address of the server from which zone transfers will
    be attempted.
    @type primary: L{str}

    @ivar _port: The port number of the server from which zone transfers will
    be attempted.
    @type _port: L{int}

    @ivar domain: The domain for which this is the secondary authority.
    @type domain: L{bytes}

    @ivar _reactor: The reactor to use to perform the zone transfers, or
    L{None} to use the global reactor.
    """
    transferring = False
    soa = records = None
    _port = 53
    _reactor = None

    def __init__(self, primaryIP, domain):
        """
        @param domain: The domain for which this will be the secondary
            authority.
        @type domain: L{bytes} or L{str}
        """
        common.ResolverBase.__init__(self)
        self.primary = nativeString(primaryIP)
        self.domain = dns.domainString(domain)

    @classmethod
    def fromServerAddressAndDomain(cls, serverAddress, domain):
        """
        Construct a new L{SecondaryAuthority} from a tuple giving a server
        address and a C{bytes} giving the name of a domain for which this is an
        authority.

        @param serverAddress: A two-tuple, the first element of which is a
            C{str} giving an IP address and the second element of which is a
            C{int} giving a port number.  Together, these define where zone
            transfers will be attempted from.

        @param domain: A C{bytes} giving the domain to transfer.
        @type domain: L{bytes}

        @return: A new instance of L{SecondaryAuthority}.
        """
        primary, port = serverAddress
        secondary = cls(primary, domain)
        secondary._port = port
        return secondary

    def transfer(self):
        """
        Attempt a zone transfer.

        @returns: A L{Deferred} that fires with L{None} when attempted zone
            transfer has completed.
        """
        if self.transferring:
            return
        self.transfering = True
        reactor = self._reactor
        if reactor is None:
            from twisted.internet import reactor
        resolver = client.Resolver(servers=[(self.primary, self._port)], reactor=reactor)
        return resolver.lookupZone(self.domain).addCallback(self._cbZone).addErrback(self._ebZone)

    def _lookup(self, name, cls, type, timeout=None):
        if not self.soa or not self.records:
            return defer.fail(failure.Failure(dns.DomainError(name)))
        return FileAuthority._lookup(self, name, cls, type, timeout)

    def _cbZone(self, zone):
        ans, _, _ = zone
        self.records = r = {}
        for rec in ans:
            if not self.soa and rec.type == dns.SOA:
                self.soa = (rec.name.name.lower(), rec.payload)
            else:
                r.setdefault(rec.name.name.lower(), []).append(rec.payload)

    def _ebZone(self, failure):
        log.msg('Updating %s from %s failed during zone transfer' % (self.domain, self.primary))
        log.err(failure)

    def update(self):
        self.transfer().addCallbacks(self._cbTransferred, self._ebTransferred)

    def _cbTransferred(self, result):
        self.transferring = False

    def _ebTransferred(self, failure):
        self.transferred = False
        log.msg('Transferring %s from %s failed after zone transfer' % (self.domain, self.primary))
        log.err(failure)