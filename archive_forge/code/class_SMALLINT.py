import datetime
from ... import exc
from ... import util
from ...sql import sqltypes
class SMALLINT(_IntegerType, sqltypes.SMALLINT):
    """MySQL SMALLINTEGER type."""
    __visit_name__ = 'SMALLINT'

    def __init__(self, display_width=None, **kw):
        """Construct a SMALLINTEGER.

        :param display_width: Optional, maximum display width for this number.

        :param unsigned: a boolean, optional.

        :param zerofill: Optional. If true, values will be stored as strings
          left-padded with zeros. Note that this does not effect the values
          returned by the underlying database API, which continue to be
          numeric.

        """
        super().__init__(display_width=display_width, **kw)