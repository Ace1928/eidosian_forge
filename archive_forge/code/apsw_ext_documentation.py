import apsw
from peewee import *
from peewee import __exception_wrapper__
from peewee import BooleanField as _BooleanField
from peewee import DateField as _DateField
from peewee import DateTimeField as _DateTimeField
from peewee import DecimalField as _DecimalField
from peewee import Insert
from peewee import TimeField as _TimeField
from peewee import logger
from playhouse.sqlite_ext import SqliteExtDatabase

Peewee integration with APSW, "another python sqlite wrapper".

Project page: https://rogerbinns.github.io/apsw/

APSW is a really neat library that provides a thin wrapper on top of SQLite's
C interface.

Here are just a few reasons to use APSW, taken from the documentation:

* APSW gives all functionality of SQLite, including virtual tables, virtual
  file system, blob i/o, backups and file control.
* Connections can be shared across threads without any additional locking.
* Transactions are managed explicitly by your code.
* APSW can handle nested transactions.
* Unicode is handled correctly.
* APSW is faster.
