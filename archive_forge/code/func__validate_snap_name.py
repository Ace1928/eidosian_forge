import errno
import re
import string
from . import exceptions as lzc_exc
from ._constants import MAXNAMELEN
def _validate_snap_name(name):
    if not _is_valid_snap_name(name):
        raise lzc_exc.SnapshotNameInvalid(name)
    elif len(name) > MAXNAMELEN:
        raise lzc_exc.NameTooLong(name)