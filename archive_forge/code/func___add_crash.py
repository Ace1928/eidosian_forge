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
def __add_crash(self, crash):
    session = self._session
    r_crash = None
    try:
        r_crash = CrashDTO(crash)
        session.add(r_crash)
        session.flush()
        crash_id = r_crash.id
    finally:
        try:
            if r_crash is not None:
                session.expire(r_crash)
        finally:
            del r_crash
    return crash_id