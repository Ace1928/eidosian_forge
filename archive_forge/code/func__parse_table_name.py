import re
from .enumerated import ENUM
from .enumerated import SET
from .types import DATETIME
from .types import TIME
from .types import TIMESTAMP
from ... import log
from ... import types as sqltypes
from ... import util
def _parse_table_name(self, line, state):
    """Extract the table name.

        :param line: The first line of SHOW CREATE TABLE
        """
    regex, cleanup = self._pr_name
    m = regex.match(line)
    if m:
        state.table_name = cleanup(m.group('name'))