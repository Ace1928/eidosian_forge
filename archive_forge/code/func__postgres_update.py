import operator
from peewee import *
from peewee import sqlite3
from peewee import Expression
from playhouse.fields import PickleField
def _postgres_update(self, __data=None, **mapping):
    if __data is not None:
        mapping.update(__data)
    return self.model.insert_many(list(mapping.items()), fields=[self.key, self.value]).on_conflict(conflict_target=[self.key], preserve=[self.value]).execute()