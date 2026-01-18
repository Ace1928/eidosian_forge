import re
from .enumerated import ENUM
from .enumerated import SET
from .types import DATETIME
from .types import TIME
from .types import TIMESTAMP
from ... import log
from ... import types as sqltypes
from ... import util
class ReflectedState:
    """Stores raw information about a SHOW CREATE TABLE statement."""

    def __init__(self):
        self.columns = []
        self.table_options = {}
        self.table_name = None
        self.keys = []
        self.fk_constraints = []
        self.ck_constraints = []