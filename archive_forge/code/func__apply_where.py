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
def _apply_where(self, query, filters, conjunction=None):
    conjunction = conjunction or operator.and_
    if filters:
        expressions = [self.model_class._meta.fields[column] == value for column, value in filters.items()]
        query = query.where(reduce(conjunction, expressions))
    return query