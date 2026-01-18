import re
from django.db.migrations.utils import get_migration_name_timestamp
from django.db.transaction import atomic
from .exceptions import IrreversibleError
class SwappableTuple(tuple):
    """
    Subclass of tuple so Django can tell this was originally a swappable
    dependency when it reads the migration file.
    """

    def __new__(cls, value, setting):
        self = tuple.__new__(cls, value)
        self.setting = setting
        return self