import os
import time
from twisted.internet import defer
from twisted.names import common, dns, error
from twisted.python import failure
from twisted.python.compat import execfile, nativeString
from twisted.python.filepath import FilePath
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