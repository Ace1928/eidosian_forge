import json
import math
import re
import struct
import sys
from peewee import *
from peewee import ColumnBase
from peewee import EnclosedNodeList
from peewee import Entity
from peewee import Expression
from peewee import Insert
from peewee import Node
from peewee import NodeList
from peewee import OP
from peewee import VirtualField
from peewee import merge_dict
from peewee import sqlite3
def _add_conn_hooks(self, conn):
    super(CSqliteExtDatabase, self)._add_conn_hooks(conn)
    self._conn_helper = ConnectionHelper(conn)
    if self._commit_hook is not None:
        self._conn_helper.set_commit_hook(self._commit_hook)
    if self._rollback_hook is not None:
        self._conn_helper.set_rollback_hook(self._rollback_hook)
    if self._update_hook is not None:
        self._conn_helper.set_update_hook(self._update_hook)
    if self._replace_busy_handler:
        timeout = self._timeout or 5
        self._conn_helper.set_busy_handler(timeout * 1000)