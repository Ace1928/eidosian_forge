import os
from typing import Dict, Type
class Psycopg2_dbutils(_PostgreSQL_dbutils):
    """Custom database utilities for Psycopg2 (PostgreSQL)."""

    def autocommit(self, conn, y=True):
        """Set autocommit on the database connection."""
        if y:
            if os.name == 'java':
                conn.autocommit = 1
            else:
                conn.set_isolation_level(0)
        elif os.name == 'java':
            conn.autocommit = 0
        else:
            conn.set_isolation_level(1)