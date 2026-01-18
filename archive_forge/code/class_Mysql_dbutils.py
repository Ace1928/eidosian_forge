import os
from typing import Dict, Type
class Mysql_dbutils(Generic_dbutils):
    """Custom database utilities for MySQL."""

    def last_id(self, cursor, table):
        """Return the last used id for a table."""
        if os.name == 'java':
            return Generic_dbutils.last_id(self, cursor, table)
        try:
            return cursor.insert_id()
        except AttributeError:
            return cursor.lastrowid