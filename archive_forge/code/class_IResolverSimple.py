from __future__ import annotations
from typing import (
from zope.interface import Attribute, Interface
from twisted.python.failure import Failure
class IResolverSimple(Interface):

    def getHostByName(name: str, timeout: Sequence[int]=()) -> 'Deferred[str]':
        """
        Resolve the domain name C{name} into an IP address.

        @param name: DNS name to resolve.
        @param timeout: Number of seconds after which to reissue the query.
            When the last timeout expires, the query is considered failed.

        @return: The callback of the Deferred that is returned will be
            passed a string that represents the IP address of the
            specified name, or the errback will be called if the
            lookup times out.  If multiple types of address records
            are associated with the name, A6 records will be returned
            in preference to AAAA records, which will be returned in
            preference to A records.  If there are multiple records of
            the type to be returned, one will be selected at random.

        @raise twisted.internet.defer.TimeoutError: Raised
            (asynchronously) if the name cannot be resolved within the
            specified timeout period.
        """