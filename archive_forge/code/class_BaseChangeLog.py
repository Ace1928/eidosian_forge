from peewee import *
from playhouse.sqlite_ext import JSONField
class BaseChangeLog(Model):
    timestamp = DateTimeField(constraints=[SQL('DEFAULT CURRENT_TIMESTAMP')])
    action = TextField()
    table = TextField()
    primary_key = IntegerField()
    changes = JSONField()