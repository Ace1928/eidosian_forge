import json
import logging
import uuid
from peewee import *
from peewee import ColumnBase
from peewee import Expression
from peewee import Node
from peewee import NodeList
from peewee import __deprecated__
class FetchManyCursor(object):
    __slots__ = ('cursor', 'array_size', 'exhausted', 'iterable')

    def __init__(self, cursor, array_size=None):
        self.cursor = cursor
        self.array_size = array_size or cursor.itersize
        self.exhausted = False
        self.iterable = self.row_gen()

    @property
    def description(self):
        return self.cursor.description

    def close(self):
        self.cursor.close()

    def row_gen(self):
        while True:
            rows = self.cursor.fetchmany(self.array_size)
            if not rows:
                return
            for row in rows:
                yield row

    def fetchone(self):
        if self.exhausted:
            return
        try:
            return next(self.iterable)
        except StopIteration:
            self.exhausted = True