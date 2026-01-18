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
def _guess_field_type(self, value):
    if isinstance(value, basestring):
        return TextField
    if isinstance(value, (datetime.date, datetime.datetime)):
        return DateTimeField
    elif value is True or value is False:
        return BooleanField
    elif isinstance(value, int):
        return IntegerField
    elif isinstance(value, float):
        return FloatField
    elif isinstance(value, Decimal):
        return DecimalField
    return TextField