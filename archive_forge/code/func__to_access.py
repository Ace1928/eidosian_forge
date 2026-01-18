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
def _to_access(self, protect):
    if protect & win32.PAGE_NOACCESS:
        access = '--- '
    elif protect & win32.PAGE_READONLY:
        access = 'R-- '
    elif protect & win32.PAGE_READWRITE:
        access = 'RW- '
    elif protect & win32.PAGE_WRITECOPY:
        access = 'RC- '
    elif protect & win32.PAGE_EXECUTE:
        access = '--X '
    elif protect & win32.PAGE_EXECUTE_READ:
        access = 'R-X '
    elif protect & win32.PAGE_EXECUTE_READWRITE:
        access = 'RWX '
    elif protect & win32.PAGE_EXECUTE_WRITECOPY:
        access = 'RCX '
    else:
        access = '??? '
    if protect & win32.PAGE_GUARD:
        access += 'G'
    else:
        access += '-'
    if protect & win32.PAGE_NOCACHE:
        access += 'N'
    else:
        access += '-'
    if protect & win32.PAGE_WRITECOMBINE:
        access += 'W'
    else:
        access += '-'
    return access