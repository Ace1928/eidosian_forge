import json
import logging
import uuid
from peewee import *
from peewee import ColumnBase
from peewee import Expression
from peewee import Node
from peewee import NodeList
from peewee import __deprecated__
class ServerSideQuery(Node):

    def __init__(self, query, array_size=None):
        self.query = query
        self.array_size = array_size
        self._cursor_wrapper = None

    def __sql__(self, ctx):
        return self.query.__sql__(ctx)

    def __iter__(self):
        if self._cursor_wrapper is None:
            self._execute(self.query._database)
        return iter(self._cursor_wrapper.iterator())

    def _execute(self, database):
        if self._cursor_wrapper is None:
            cursor = database.execute(self.query, named_cursor=True, array_size=self.array_size)
            self._cursor_wrapper = self.query._get_cursor_wrapper(cursor)
        return self._cursor_wrapper