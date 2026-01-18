import operator
from peewee import *
from peewee import sqlite3
from peewee import Expression
from playhouse.fields import PickleField
def _postgres_upsert(self, key, value):
    self.model.insert(key=key, value=value).on_conflict(conflict_target=[self.key], preserve=[self.value]).execute()