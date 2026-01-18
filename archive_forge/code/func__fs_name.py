import errno
import re
import string
from . import exceptions as lzc_exc
from ._constants import MAXNAMELEN
def _fs_name(name):
    """
    Extract a dataset name from the given snapshot or bookmark name.

    '@' separates a snapshot name from the rest of the dataset name.
    '#' separates a bookmark name from the rest of the dataset name.
    """
    return re.split('[@#]', name, 1)[0]