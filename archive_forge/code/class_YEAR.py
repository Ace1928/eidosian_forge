import datetime
from ... import exc
from ... import util
from ...sql import sqltypes
class YEAR(sqltypes.TypeEngine):
    """MySQL YEAR type, for single byte storage of years 1901-2155."""
    __visit_name__ = 'YEAR'

    def __init__(self, display_width=None):
        self.display_width = display_width