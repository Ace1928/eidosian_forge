import csv
import datetime
from decimal import Decimal
import json
import operator
import sys
import uuid
from peewee import *
from playhouse.db_url import connect
from playhouse.migrate import migrate
from playhouse.migrate import SchemaMigrator
from playhouse.reflection import Introspector
def _migrate_new_columns(self, data):
    new_keys = set(data) - set(self.model_class._meta.fields)
    new_keys -= set(self.model_class._meta.columns)
    if new_keys:
        operations = []
        for key in new_keys:
            field_class = self._guess_field_type(data[key])
            field = field_class(null=True)
            operations.append(self.dataset._migrator.add_column(self.name, key, field))
            field.bind(self.model_class, key)
        migrate(*operations)
        self.dataset.update_cache(self.name)