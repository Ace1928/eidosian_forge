import datetime
import decimal
import uuid
from functools import lru_cache
from itertools import chain
from django.conf import settings
from django.core.exceptions import FieldError
from django.db import DatabaseError, NotSupportedError, models
from django.db.backends.base.operations import BaseDatabaseOperations
from django.db.models.constants import OnConflict
from django.db.models.expressions import Col
from django.utils import timezone
from django.utils.dateparse import parse_date, parse_datetime, parse_time
from django.utils.functional import cached_property
def _quote_params_for_last_executed_query(self, params):
    """
        Only for last_executed_query! Don't use this to execute SQL queries!
        """
    BATCH_SIZE = 999
    if len(params) > BATCH_SIZE:
        results = ()
        for index in range(0, len(params), BATCH_SIZE):
            chunk = params[index:index + BATCH_SIZE]
            results += self._quote_params_for_last_executed_query(chunk)
        return results
    sql = 'SELECT ' + ', '.join(['QUOTE(?)'] * len(params))
    cursor = self.connection.connection.cursor()
    try:
        return cursor.execute(sql, params).fetchone()
    finally:
        cursor.close()