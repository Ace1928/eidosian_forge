from io import BytesIO
import struct
import random
import time
import dns.exception
import dns.tsig
from ._compat import long
def _rollback(self, where):
    """Truncate the output buffer at offset *where*, and remove any
        compression table entries that pointed beyond the truncation
        point.
        """
    self.output.seek(where)
    self.output.truncate()
    keys_to_delete = []
    for k, v in self.compress.items():
        if v >= where:
            keys_to_delete.append(k)
    for k in keys_to_delete:
        del self.compress[k]