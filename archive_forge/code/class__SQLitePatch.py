import sqlite3
import datetime
import warnings
from sqlalchemy import create_engine, Column, ForeignKey, Sequence
from sqlalchemy.engine.url import URL
from sqlalchemy.ext.compiler import compiles
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.interfaces import PoolListener
from sqlalchemy.orm import sessionmaker, deferred
from sqlalchemy.orm.exc import NoResultFound, MultipleResultsFound
from sqlalchemy.types import Integer, BigInteger, Boolean, DateTime, String, \
from sqlalchemy.sql.expression import asc, desc
from crash import Crash, Marshaller, pickle, HIGHEST_PROTOCOL
from textio import CrashDump
import win32
class _SQLitePatch(PoolListener):
    """
    Used internally by L{BaseDAO}.

    After connecting to an SQLite database, ensure that the foreign keys
    support is enabled. If not, abort the connection.

    @see: U{http://sqlite.org/foreignkeys.html}
    """

    def connect(dbapi_connection, connection_record):
        """
        Called once by SQLAlchemy for each new SQLite DB-API connection.

        Here is where we issue some PRAGMA statements to configure how we're
        going to access the SQLite database.

        @param dbapi_connection:
            A newly connected raw SQLite DB-API connection.

        @param connection_record:
            Unused by this method.
        """
        try:
            cursor = dbapi_connection.cursor()
            try:
                cursor.execute('PRAGMA foreign_keys = ON;')
                cursor.execute('PRAGMA foreign_keys;')
                if cursor.fetchone()[0] != 1:
                    raise Exception()
            finally:
                cursor.close()
        except Exception:
            dbapi_connection.close()
            raise sqlite3.Error()