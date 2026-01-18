from gitdb.util import (
from gitdb.utils.encoding import force_text
from gitdb.exc import (
from itertools import chain
from functools import reduce
def _databases_recursive(database, output):
    """Fill output list with database from db, in order. Deals with Loose, Packed
    and compound databases."""
    if isinstance(database, CompoundDB):
        dbs = database.databases()
        output.extend((db for db in dbs if not isinstance(db, CompoundDB)))
        for cdb in (db for db in dbs if isinstance(db, CompoundDB)):
            _databases_recursive(cdb, output)
    else:
        output.append(database)