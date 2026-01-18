import re
import sys
import ovs.db.parser
import ovs.db.types
from ovs.db import error
def __root_set_size(self):
    """Returns the number of tables in the schema's root set."""
    n_root = 0
    for table in self.tables.values():
        if table.is_root:
            n_root += 1
    return n_root