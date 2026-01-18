import datetime
from ... import exc
from ... import util
from ...sql import sqltypes
class TINYBLOB(sqltypes._Binary):
    """MySQL TINYBLOB type, for binary data up to 2^8 bytes."""
    __visit_name__ = 'TINYBLOB'