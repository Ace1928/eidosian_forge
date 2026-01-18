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
def _check_sql(self, name, check):
    return self.sql_constraint % {'name': self.quote_name(name), 'constraint': self.sql_check_constraint % {'check': check}}