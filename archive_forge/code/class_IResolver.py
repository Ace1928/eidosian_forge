from __future__ import annotations
from typing import (
from zope.interface import Attribute, Interface
from twisted.python.failure import Failure
class IResolver(IResolverSimple):

    def query(query: 'Query', timeout: Sequence[int]) -> 'Deferred[Tuple[RRHeader, RRHeader, RRHeader]]':
        """
        Dispatch C{query} to the method which can handle its type.

        @param query: The DNS query being issued, to which a response is to be
            generated.
        @param timeout: Number of seconds after which to reissue the query.
            When the last timeout expires, the query is considered failed.

        @return: A L{Deferred} which fires with a three-tuple of lists of
            L{twisted.names.dns.RRHeader} instances.  The first element of the
            tuple gives answers.  The second element of the tuple gives
            authorities.  The third element of the tuple gives additional
            information.  The L{Deferred} may instead fail with one of the
            exceptions defined in L{twisted.names.error} or with
            C{NotImplementedError}.
        """

    def lookupAddress(name: str, timeout: Sequence[int]) -> 'Deferred[Tuple[RRHeader, RRHeader, RRHeader]]':
        """
        Perform an A record lookup.

        @param name: DNS name to resolve.
        @param timeout: Number of seconds after which to reissue the query.
            When the last timeout expires, the query is considered failed.

        @return: A L{Deferred} which fires with a three-tuple of lists of
            L{twisted.names.dns.RRHeader} instances.  The first element of the
            tuple gives answers.  The second element of the tuple gives
            authorities.  The third element of the tuple gives additional
            information.  The L{Deferred} may instead fail with one of the
            exceptions defined in L{twisted.names.error} or with
            C{NotImplementedError}.
        """

    def lookupAddress6(name: str, timeout: Sequence[int]) -> 'Deferred[Tuple[RRHeader, RRHeader, RRHeader]]':
        """
        Perform an A6 record lookup.

        @param name: DNS name to resolve.
        @param timeout: Number of seconds after which to reissue the query.
            When the last timeout expires, the query is considered failed.

        @return: A L{Deferred} which fires with a three-tuple of lists of
            L{twisted.names.dns.RRHeader} instances.  The first element of the
            tuple gives answers.  The second element of the tuple gives
            authorities.  The third element of the tuple gives additional
            information.  The L{Deferred} may instead fail with one of the
            exceptions defined in L{twisted.names.error} or with
            C{NotImplementedError}.
        """

    def lookupIPV6Address(name: str, timeout: Sequence[int]) -> 'Deferred[Tuple[RRHeader, RRHeader, RRHeader]]':
        """
        Perform an AAAA record lookup.

        @param name: DNS name to resolve.
        @param timeout: Number of seconds after which to reissue the query.
            When the last timeout expires, the query is considered failed.

        @return: A L{Deferred} which fires with a three-tuple of lists of
            L{twisted.names.dns.RRHeader} instances.  The first element of the
            tuple gives answers.  The second element of the tuple gives
            authorities.  The third element of the tuple gives additional
            information.  The L{Deferred} may instead fail with one of the
            exceptions defined in L{twisted.names.error} or with
            C{NotImplementedError}.
        """

    def lookupMailExchange(name: str, timeout: Sequence[int]) -> 'Deferred[Tuple[RRHeader, RRHeader, RRHeader]]':
        """
        Perform an MX record lookup.

        @param name: DNS name to resolve.
        @param timeout: Number of seconds after which to reissue the query.
            When the last timeout expires, the query is considered failed.

        @return: A L{Deferred} which fires with a three-tuple of lists of
            L{twisted.names.dns.RRHeader} instances.  The first element of the
            tuple gives answers.  The second element of the tuple gives
            authorities.  The third element of the tuple gives additional
            information.  The L{Deferred} may instead fail with one of the
            exceptions defined in L{twisted.names.error} or with
            C{NotImplementedError}.
        """

    def lookupNameservers(name: str, timeout: Sequence[int]) -> 'Deferred[Tuple[RRHeader, RRHeader, RRHeader]]':
        """
        Perform an NS record lookup.

        @param name: DNS name to resolve.
        @param timeout: Number of seconds after which to reissue the query.
            When the last timeout expires, the query is considered failed.

        @return: A L{Deferred} which fires with a three-tuple of lists of
            L{twisted.names.dns.RRHeader} instances.  The first element of the
            tuple gives answers.  The second element of the tuple gives
            authorities.  The third element of the tuple gives additional
            information.  The L{Deferred} may instead fail with one of the
            exceptions defined in L{twisted.names.error} or with
            C{NotImplementedError}.
        """

    def lookupCanonicalName(name: str, timeout: Sequence[int]) -> 'Deferred[Tuple[RRHeader, RRHeader, RRHeader]]':
        """
        Perform a CNAME record lookup.

        @param name: DNS name to resolve.
        @param timeout: Number of seconds after which to reissue the query.
            When the last timeout expires, the query is considered failed.

        @return: A L{Deferred} which fires with a three-tuple of lists of
            L{twisted.names.dns.RRHeader} instances.  The first element of the
            tuple gives answers.  The second element of the tuple gives
            authorities.  The third element of the tuple gives additional
            information.  The L{Deferred} may instead fail with one of the
            exceptions defined in L{twisted.names.error} or with
            C{NotImplementedError}.
        """

    def lookupMailBox(name: str, timeout: Sequence[int]) -> 'Deferred[Tuple[RRHeader, RRHeader, RRHeader]]':
        """
        Perform an MB record lookup.

        @param name: DNS name to resolve.
        @param timeout: Number of seconds after which to reissue the query.
            When the last timeout expires, the query is considered failed.

        @return: A L{Deferred} which fires with a three-tuple of lists of
            L{twisted.names.dns.RRHeader} instances.  The first element of the
            tuple gives answers.  The second element of the tuple gives
            authorities.  The third element of the tuple gives additional
            information.  The L{Deferred} may instead fail with one of the
            exceptions defined in L{twisted.names.error} or with
            C{NotImplementedError}.
        """

    def lookupMailGroup(name: str, timeout: Sequence[int]) -> 'Deferred[Tuple[RRHeader, RRHeader, RRHeader]]':
        """
        Perform an MG record lookup.

        @param name: DNS name to resolve.
        @param timeout: Number of seconds after which to reissue the query.
            When the last timeout expires, the query is considered failed.

        @return: A L{Deferred} which fires with a three-tuple of lists of
            L{twisted.names.dns.RRHeader} instances.  The first element of the
            tuple gives answers.  The second element of the tuple gives
            authorities.  The third element of the tuple gives additional
            information.  The L{Deferred} may instead fail with one of the
            exceptions defined in L{twisted.names.error} or with
            C{NotImplementedError}.
        """

    def lookupMailRename(name: str, timeout: Sequence[int]) -> 'Deferred[Tuple[RRHeader, RRHeader, RRHeader]]':
        """
        Perform an MR record lookup.

        @param name: DNS name to resolve.
        @param timeout: Number of seconds after which to reissue the query.
            When the last timeout expires, the query is considered failed.

        @return: A L{Deferred} which fires with a three-tuple of lists of
            L{twisted.names.dns.RRHeader} instances.  The first element of the
            tuple gives answers.  The second element of the tuple gives
            authorities.  The third element of the tuple gives additional
            information.  The L{Deferred} may instead fail with one of the
            exceptions defined in L{twisted.names.error} or with
            C{NotImplementedError}.
        """

    def lookupPointer(name: str, timeout: Sequence[int]) -> 'Deferred[Tuple[RRHeader, RRHeader, RRHeader]]':
        """
        Perform a PTR record lookup.

        @param name: DNS name to resolve.
        @param timeout: Number of seconds after which to reissue the query.
            When the last timeout expires, the query is considered failed.

        @return: A L{Deferred} which fires with a three-tuple of lists of
            L{twisted.names.dns.RRHeader} instances.  The first element of the
            tuple gives answers.  The second element of the tuple gives
            authorities.  The third element of the tuple gives additional
            information.  The L{Deferred} may instead fail with one of the
            exceptions defined in L{twisted.names.error} or with
            C{NotImplementedError}.
        """

    def lookupAuthority(name: str, timeout: Sequence[int]) -> 'Deferred[Tuple[RRHeader, RRHeader, RRHeader]]':
        """
        Perform an SOA record lookup.

        @param name: DNS name to resolve.
        @param timeout: Number of seconds after which to reissue the query.
            When the last timeout expires, the query is considered failed.

        @return: A L{Deferred} which fires with a three-tuple of lists of
            L{twisted.names.dns.RRHeader} instances.  The first element of the
            tuple gives answers.  The second element of the tuple gives
            authorities.  The third element of the tuple gives additional
            information.  The L{Deferred} may instead fail with one of the
            exceptions defined in L{twisted.names.error} or with
            C{NotImplementedError}.
        """

    def lookupNull(name: str, timeout: Sequence[int]) -> 'Deferred[Tuple[RRHeader, RRHeader, RRHeader]]':
        """
        Perform a NULL record lookup.

        @param name: DNS name to resolve.
        @param timeout: Number of seconds after which to reissue the query.
            When the last timeout expires, the query is considered failed.

        @return: A L{Deferred} which fires with a three-tuple of lists of
            L{twisted.names.dns.RRHeader} instances.  The first element of the
            tuple gives answers.  The second element of the tuple gives
            authorities.  The third element of the tuple gives additional
            information.  The L{Deferred} may instead fail with one of the
            exceptions defined in L{twisted.names.error} or with
            C{NotImplementedError}.
        """

    def lookupWellKnownServices(name: str, timeout: Sequence[int]) -> 'Deferred[Tuple[RRHeader, RRHeader, RRHeader]]':
        """
        Perform a WKS record lookup.

        @param name: DNS name to resolve.
        @param timeout: Number of seconds after which to reissue the query.
            When the last timeout expires, the query is considered failed.

        @return: A L{Deferred} which fires with a three-tuple of lists of
            L{twisted.names.dns.RRHeader} instances.  The first element of the
            tuple gives answers.  The second element of the tuple gives
            authorities.  The third element of the tuple gives additional
            information.  The L{Deferred} may instead fail with one of the
            exceptions defined in L{twisted.names.error} or with
            C{NotImplementedError}.
        """

    def lookupHostInfo(name: str, timeout: Sequence[int]) -> 'Deferred[Tuple[RRHeader, RRHeader, RRHeader]]':
        """
        Perform a HINFO record lookup.

        @param name: DNS name to resolve.
        @param timeout: Number of seconds after which to reissue the query.
            When the last timeout expires, the query is considered failed.

        @return: A L{Deferred} which fires with a three-tuple of lists of
            L{twisted.names.dns.RRHeader} instances.  The first element of the
            tuple gives answers.  The second element of the tuple gives
            authorities.  The third element of the tuple gives additional
            information.  The L{Deferred} may instead fail with one of the
            exceptions defined in L{twisted.names.error} or with
            C{NotImplementedError}.
        """

    def lookupMailboxInfo(name: str, timeout: Sequence[int]) -> 'Deferred[Tuple[RRHeader, RRHeader, RRHeader]]':
        """
        Perform an MINFO record lookup.

        @param name: DNS name to resolve.
        @param timeout: Number of seconds after which to reissue the query.
            When the last timeout expires, the query is considered failed.

        @return: A L{Deferred} which fires with a three-tuple of lists of
            L{twisted.names.dns.RRHeader} instances.  The first element of the
            tuple gives answers.  The second element of the tuple gives
            authorities.  The third element of the tuple gives additional
            information.  The L{Deferred} may instead fail with one of the
            exceptions defined in L{twisted.names.error} or with
            C{NotImplementedError}.
        """

    def lookupText(name: str, timeout: Sequence[int]) -> 'Deferred[Tuple[RRHeader, RRHeader, RRHeader]]':
        """
        Perform a TXT record lookup.

        @param name: DNS name to resolve.
        @param timeout: Number of seconds after which to reissue the query.
            When the last timeout expires, the query is considered failed.

        @return: A L{Deferred} which fires with a three-tuple of lists of
            L{twisted.names.dns.RRHeader} instances.  The first element of the
            tuple gives answers.  The second element of the tuple gives
            authorities.  The third element of the tuple gives additional
            information.  The L{Deferred} may instead fail with one of the
            exceptions defined in L{twisted.names.error} or with
            C{NotImplementedError}.
        """

    def lookupResponsibility(name: str, timeout: Sequence[int]) -> 'Deferred[Tuple[RRHeader, RRHeader, RRHeader]]':
        """
        Perform an RP record lookup.

        @param name: DNS name to resolve.
        @param timeout: Number of seconds after which to reissue the query.
            When the last timeout expires, the query is considered failed.

        @return: A L{Deferred} which fires with a three-tuple of lists of
            L{twisted.names.dns.RRHeader} instances.  The first element of the
            tuple gives answers.  The second element of the tuple gives
            authorities.  The third element of the tuple gives additional
            information.  The L{Deferred} may instead fail with one of the
            exceptions defined in L{twisted.names.error} or with
            C{NotImplementedError}.
        """

    def lookupAFSDatabase(name: str, timeout: Sequence[int]) -> 'Deferred[Tuple[RRHeader, RRHeader, RRHeader]]':
        """
        Perform an AFSDB record lookup.

        @param name: DNS name to resolve.
        @param timeout: Number of seconds after which to reissue the query.
            When the last timeout expires, the query is considered failed.

        @return: A L{Deferred} which fires with a three-tuple of lists of
            L{twisted.names.dns.RRHeader} instances.  The first element of the
            tuple gives answers.  The second element of the tuple gives
            authorities.  The third element of the tuple gives additional
            information.  The L{Deferred} may instead fail with one of the
            exceptions defined in L{twisted.names.error} or with
            C{NotImplementedError}.
        """

    def lookupService(name: str, timeout: Sequence[int]) -> 'Deferred[Tuple[RRHeader, RRHeader, RRHeader]]':
        """
        Perform an SRV record lookup.

        @param name: DNS name to resolve.
        @param timeout: Number of seconds after which to reissue the query.
            When the last timeout expires, the query is considered failed.

        @return: A L{Deferred} which fires with a three-tuple of lists of
            L{twisted.names.dns.RRHeader} instances.  The first element of the
            tuple gives answers.  The second element of the tuple gives
            authorities.  The third element of the tuple gives additional
            information.  The L{Deferred} may instead fail with one of the
            exceptions defined in L{twisted.names.error} or with
            C{NotImplementedError}.
        """

    def lookupAllRecords(name: str, timeout: Sequence[int]) -> 'Deferred[Tuple[RRHeader, RRHeader, RRHeader]]':
        """
        Perform an ALL_RECORD lookup.

        @param name: DNS name to resolve.
        @param timeout: Number of seconds after which to reissue the query.
            When the last timeout expires, the query is considered failed.

        @return: A L{Deferred} which fires with a three-tuple of lists of
            L{twisted.names.dns.RRHeader} instances.  The first element of the
            tuple gives answers.  The second element of the tuple gives
            authorities.  The third element of the tuple gives additional
            information.  The L{Deferred} may instead fail with one of the
            exceptions defined in L{twisted.names.error} or with
            C{NotImplementedError}.
        """

    def lookupSenderPolicy(name: str, timeout: Sequence[int]) -> 'Deferred[Tuple[RRHeader, RRHeader, RRHeader]]':
        """
        Perform a SPF record lookup.

        @param name: DNS name to resolve.
        @param timeout: Number of seconds after which to reissue the query.
            When the last timeout expires, the query is considered failed.

        @return: A L{Deferred} which fires with a three-tuple of lists of
            L{twisted.names.dns.RRHeader} instances.  The first element of the
            tuple gives answers.  The second element of the tuple gives
            authorities.  The third element of the tuple gives additional
            information.  The L{Deferred} may instead fail with one of the
            exceptions defined in L{twisted.names.error} or with
            C{NotImplementedError}.
        """

    def lookupNamingAuthorityPointer(name: str, timeout: Sequence[int]) -> 'Deferred[Tuple[RRHeader, RRHeader, RRHeader]]':
        """
        Perform a NAPTR record lookup.

        @param name: DNS name to resolve.
        @param timeout: Number of seconds after which to reissue the query.
            When the last timeout expires, the query is considered failed.

        @return: A L{Deferred} which fires with a three-tuple of lists of
            L{twisted.names.dns.RRHeader} instances.  The first element of the
            tuple gives answers.  The second element of the tuple gives
            authorities.  The third element of the tuple gives additional
            information.  The L{Deferred} may instead fail with one of the
            exceptions defined in L{twisted.names.error} or with
            C{NotImplementedError}.
        """

    def lookupZone(name: str, timeout: Sequence[int]) -> 'Deferred[Tuple[RRHeader, RRHeader, RRHeader]]':
        """
        Perform an AXFR record lookup.

        NB This is quite different from other DNS requests. See
        U{http://cr.yp.to/djbdns/axfr-notes.html} for more
        information.

        NB Unlike other C{lookup*} methods, the timeout here is not a
        list of ints, it is a single int.

        @param name: DNS name to resolve.
        @param timeout: When this timeout expires, the query is
            considered failed.

        @return: A L{Deferred} which fires with a three-tuple of lists of
            L{twisted.names.dns.RRHeader} instances.
            The first element of the tuple gives answers.
            The second and third elements are always empty.
            The L{Deferred} may instead fail with one of the
            exceptions defined in L{twisted.names.error} or with
            C{NotImplementedError}.
        """