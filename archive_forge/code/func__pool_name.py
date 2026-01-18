import errno
import re
import string
from . import exceptions as lzc_exc
from ._constants import MAXNAMELEN
def _pool_name(name):
    """
    Extract a pool name from the given dataset or bookmark name.

    '/' separates dataset name components.
    '@' separates a snapshot name from the rest of the dataset name.
    '#' separates a bookmark name from the rest of the dataset name.
    """
    return re.split('[/@#]', name, 1)[0]