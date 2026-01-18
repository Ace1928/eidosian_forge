import os
import time
from twisted.internet import defer
from twisted.names import common, dns, error
from twisted.python import failure
from twisted.python.compat import execfile, nativeString
from twisted.python.filepath import FilePath

        Parse a C{line} from a zone file respecting C{origin} and C{ttl}.

        Add resulting records to authority.

        @param origin: starting point for the zone
        @type origin: L{bytes}

        @param ttl: time to live for the record
        @type ttl: L{int}

        @param line: zone file line to parse; split by word
        @type line: L{list} of L{bytes}
        