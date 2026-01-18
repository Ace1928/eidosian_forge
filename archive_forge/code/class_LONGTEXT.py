import datetime
from ... import exc
from ... import util
from ...sql import sqltypes
class LONGTEXT(_StringType):
    """MySQL LONGTEXT type, for character storage encoded up to 2^32 bytes."""
    __visit_name__ = 'LONGTEXT'

    def __init__(self, **kwargs):
        """Construct a LONGTEXT.

        :param charset: Optional, a column-level character set for this string
          value.  Takes precedence to 'ascii' or 'unicode' short-hand.

        :param collation: Optional, a column-level collation for this string
          value.  Takes precedence to 'binary' short-hand.

        :param ascii: Defaults to False: short-hand for the ``latin1``
          character set, generates ASCII in schema.

        :param unicode: Defaults to False: short-hand for the ``ucs2``
          character set, generates UNICODE in schema.

        :param national: Optional. If true, use the server's configured
          national character set.

        :param binary: Defaults to False: short-hand, pick the binary
          collation type that matches the column's character set.  Generates
          BINARY in schema.  This does not affect the type of data stored,
          only the collation of character data.

        """
        super().__init__(**kwargs)