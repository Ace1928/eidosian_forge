from ..lazy_import import lazy_import
import time
from breezy import (
from .. import lazy_regex
def _next_id_suffix():
    """Create a new file id suffix that is reasonably unique.

    On the first call we combine the current time with 64 bits of randomness to
    give a highly probably globally unique number. Then each call in the same
    process adds 1 to a serial number we append to that unique value.
    """
    global _gen_file_id_suffix, _gen_file_id_serial
    if _gen_file_id_suffix is None:
        _gen_file_id_suffix = '-{}-{}-'.format(osutils.compact_date(time.time()), osutils.rand_chars(16)).encode('ascii')
    _gen_file_id_serial += 1
    return b'%s%d' % (_gen_file_id_suffix, _gen_file_id_serial)