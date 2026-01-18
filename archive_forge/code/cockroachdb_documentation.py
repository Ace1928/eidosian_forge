import functools
import re
import sys
from peewee import *
from peewee import _atomic
from peewee import _manual
from peewee import ColumnMetadata  # (name, data_type, null, primary_key, table, default)
from peewee import EnclosedNodeList
from peewee import Entity
from peewee import ForeignKeyMetadata  # (column, dest_table, dest_column, table).
from peewee import IndexMetadata
from peewee import NodeList
from playhouse.pool import _PooledPostgresqlDatabase

    Run transactional SQL in a transaction with automatic retries.

    User-provided `callback`:
    * Must accept one parameter, the `db` instance representing the connection
      the transaction is running under.
    * Must not attempt to commit, rollback or otherwise manage transactions.
    * May be called more than once.
    * Should ideally only contain SQL operations.

    Additionally, the database must not have any open transaction at the time
    this function is called, as CRDB does not support nested transactions.
    