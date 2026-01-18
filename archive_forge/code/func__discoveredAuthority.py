from twisted.internet import defer
from twisted.names import common, dns, error
from twisted.python.failure import Failure
def _discoveredAuthority(self, response, query, timeout, queriesLeft):
    """
        Interpret the response to a query, checking for error codes and
        following delegations if necessary.

        @param response: The L{Message} received in response to issuing C{query}.
        @type response: L{Message}

        @param query: The L{dns.Query} which was issued.
        @type query: L{dns.Query}.

        @param timeout: The timeout to use if another query is indicated by
            this response.
        @type timeout: L{tuple} of L{int}

        @param queriesLeft: A C{int} giving the number of queries which may
            yet be attempted to answer this query before the attempt will be
            abandoned.

        @return: A L{Failure} indicating a response error, a three-tuple of
            lists of L{twisted.names.dns.RRHeader} giving the response to
            C{query} or a L{Deferred} which will fire with one of those.
        """
    if response.rCode != dns.OK:
        return Failure(self.exceptionForCode(response.rCode)(response))
    records = {}
    for answer in response.answers:
        records.setdefault(answer.name, []).append(answer)

    def findAnswerOrCName(name, type, cls):
        cname = None
        for record in records.get(name, []):
            if record.cls == cls:
                if record.type == type:
                    return record
                elif record.type == dns.CNAME:
                    cname = record
        return cname
    seen = set()
    name = query.name
    record = None
    while True:
        seen.add(name)
        previous = record
        record = findAnswerOrCName(name, query.type, query.cls)
        if record is None:
            if name == query.name:
                break
            else:
                d = self._discoverAuthority(dns.Query(str(name), query.type, query.cls), self._roots(), timeout, queriesLeft)

                def cbResolved(results):
                    answers, authority, additional = results
                    answers.insert(0, previous)
                    return (answers, authority, additional)
                d.addCallback(cbResolved)
                return d
        elif record.type == query.type:
            return (response.answers, response.authority, response.additional)
        else:
            if record.payload.name in seen:
                raise error.ResolverError('Cycle in CNAME processing')
            name = record.payload.name
    addresses = {}
    for rr in response.additional:
        if rr.type == dns.A:
            addresses[rr.name.name] = rr.payload.dottedQuad()
    hints = []
    traps = []
    for rr in response.authority:
        if rr.type == dns.NS:
            ns = rr.payload.name.name
            if ns in addresses:
                hints.append((addresses[ns], dns.PORT))
            else:
                traps.append(ns)
    if hints:
        return self._discoverAuthority(query, hints, timeout, queriesLeft)
    elif traps:
        d = self.lookupAddress(traps[0], timeout)

        def getOneAddress(results):
            answers, authority, additional = results
            return answers[0].payload.dottedQuad()
        d.addCallback(getOneAddress)
        d.addCallback(lambda hint: self._discoverAuthority(query, [(hint, dns.PORT)], timeout, queriesLeft - 1))
        return d
    else:
        return Failure(error.ResolverError('Stuck at response without answers or delegation'))