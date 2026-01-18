import warnings
from django.db.models import CharField, EmailField, TextField
from django.test.utils import ignore_warnings
from django.utils.deprecation import RemovedInDjango51Warning
class CIEmailField(CIText, EmailField):
    system_check_deprecated_details = {'msg': 'django.contrib.postgres.fields.CIEmailField is deprecated. Support for it (except in historical migrations) will be removed in Django 5.1.', 'hint': 'Use EmailField(db_collation="…") with a case-insensitive non-deterministic collation instead.', 'id': 'fields.W906'}

    def __init__(self, *args, **kwargs):
        with ignore_warnings(category=RemovedInDjango51Warning):
            super().__init__(*args, **kwargs)