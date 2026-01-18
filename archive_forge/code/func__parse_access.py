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
@staticmethod
def _parse_access(access):
    if not access:
        return 0
    perm = access[:3]
    if perm == 'R--':
        protect = win32.PAGE_READONLY
    elif perm == 'RW-':
        protect = win32.PAGE_READWRITE
    elif perm == 'RC-':
        protect = win32.PAGE_WRITECOPY
    elif perm == '--X':
        protect = win32.PAGE_EXECUTE
    elif perm == 'R-X':
        protect = win32.PAGE_EXECUTE_READ
    elif perm == 'RWX':
        protect = win32.PAGE_EXECUTE_READWRITE
    elif perm == 'RCX':
        protect = win32.PAGE_EXECUTE_WRITECOPY
    else:
        protect = win32.PAGE_NOACCESS
    if access[5] == 'G':
        protect = protect | win32.PAGE_GUARD
    if access[6] == 'N':
        protect = protect | win32.PAGE_NOCACHE
    if access[7] == 'W':
        protect = protect | win32.PAGE_WRITECOMBINE
    return protect