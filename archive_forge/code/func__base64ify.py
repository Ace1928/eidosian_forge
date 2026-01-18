from io import BytesIO
import base64
import binascii
import dns.exception
import dns.name
import dns.rdataclass
import dns.rdatatype
import dns.tokenizer
import dns.wiredata
from ._compat import xrange, string_types, text_type
def _base64ify(data, chunksize=_base64_chunksize):
    """Convert a binary string into its base64 encoding, broken up into chunks
    of chunksize characters separated by a space.
    """
    line = base64.b64encode(data)
    return b' '.join([line[i:i + chunksize] for i in range(0, len(line), chunksize)]).decode()