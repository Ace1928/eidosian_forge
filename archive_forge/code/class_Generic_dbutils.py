import os
from typing import Dict, Type
class Generic_dbutils:
    """Default database utilities."""

    def __init__(self):
        """Create a Generic_dbutils object."""

    def tname(self, table):
        """Return the name of the table."""
        if table != 'biosequence':
            return table
        else:
            return 'bioentry'

    def last_id(self, cursor, table):
        """Return the last used id for a table."""
        table = self.tname(table)
        sql = f'select max({table}_id) from {table}'
        cursor.execute(sql)
        rv = cursor.fetchone()
        return rv[0]

    def execute(self, cursor, sql, args=None):
        """Just execute an sql command."""
        cursor.execute(sql, args or ())

    def executemany(self, cursor, sql, seq):
        """Execute many sql commands."""
        cursor.executemany(sql, seq)

    def autocommit(self, conn, y=1):
        """Set autocommit on the database connection."""