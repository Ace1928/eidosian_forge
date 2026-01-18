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
def _model_indexes_sql(self, model):
    """
        Return a list of all index SQL statements (field indexes,
        index_together, Meta.indexes) for the specified model.
        """
    if not model._meta.managed or model._meta.proxy or model._meta.swapped:
        return []
    output = []
    for field in model._meta.local_fields:
        output.extend(self._field_indexes_sql(model, field))
    for field_names in model._meta.index_together:
        fields = [model._meta.get_field(field) for field in field_names]
        output.append(self._create_index_sql(model, fields=fields, suffix='_idx'))
    for index in model._meta.indexes:
        if not index.contains_expressions or self.connection.features.supports_expression_indexes:
            output.append(index.create_sql(model, self))
    return output