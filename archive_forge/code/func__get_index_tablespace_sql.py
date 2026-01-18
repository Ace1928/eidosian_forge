import logging
import operator
from datetime import datetime
from django.conf import settings
from django.db.backends.ddl_references import (
from django.db.backends.utils import names_digest, split_identifier, truncate_name
from django.db.models import NOT_PROVIDED, Deferrable, Index
from django.db.models.sql import Query
from django.db.transaction import TransactionManagementError, atomic
from django.utils import timezone
def _get_index_tablespace_sql(self, model, fields, db_tablespace=None):
    if db_tablespace is None:
        if len(fields) == 1 and fields[0].db_tablespace:
            db_tablespace = fields[0].db_tablespace
        elif settings.DEFAULT_INDEX_TABLESPACE:
            db_tablespace = settings.DEFAULT_INDEX_TABLESPACE
        elif model._meta.db_tablespace:
            db_tablespace = model._meta.db_tablespace
    if db_tablespace is not None:
        return ' ' + self.connection.ops.tablespace_sql(db_tablespace)
    return ''