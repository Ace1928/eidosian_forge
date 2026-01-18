from django.contrib.postgres.signals import (
from django.db import NotSupportedError, router
from django.db.migrations import AddConstraint, AddIndex, RemoveIndex
from django.db.migrations.operations.base import Operation
from django.db.models.constraints import CheckConstraint
class NotInTransactionMixin:

    def _ensure_not_in_transaction(self, schema_editor):
        if schema_editor.connection.in_atomic_block:
            raise NotSupportedError('The %s operation cannot be executed inside a transaction (set atomic = False on the migration).' % self.__class__.__name__)