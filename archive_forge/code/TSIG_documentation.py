import base64
import struct
import dns.exception
import dns.immutable
import dns.rcode
import dns.rdata
Initialize a TSIG rdata.

        *rdclass*, an ``int`` is the rdataclass of the Rdata.

        *rdtype*, an ``int`` is the rdatatype of the Rdata.

        *algorithm*, a ``dns.name.Name``.

        *time_signed*, an ``int``.

        *fudge*, an ``int`.

        *mac*, a ``bytes``

        *original_id*, an ``int``

        *error*, an ``int``

        *other*, a ``bytes``
        