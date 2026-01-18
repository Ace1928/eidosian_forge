import datetime
from ... import exc
from ... import util
from ...sql import sqltypes
class LONGBLOB(sqltypes._Binary):
    """MySQL LONGBLOB type, for binary data up to 2^32 bytes."""
    __visit_name__ = 'LONGBLOB'