import re
import sys
import ovs.db.parser
import ovs.db.types
from ovs.db import error
def __follow_ref_table(self, column, base, base_name):
    if not base or base.type != ovs.db.types.UuidType or (not base.ref_table_name):
        return
    base.ref_table = self.tables.get(base.ref_table_name)
    if not base.ref_table:
        raise error.Error('column %s %s refers to undefined table %s' % (column.name, base_name, base.ref_table_name), tag='syntax error')
    if base.is_strong_ref() and (not base.ref_table.is_root):
        column.persistent = True