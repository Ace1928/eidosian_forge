from peewee import *
from playhouse.sqlite_ext import JSONField
class ChangeLog(self.base_model):

    class Meta:
        database = self.db
        table_name = self.table_name