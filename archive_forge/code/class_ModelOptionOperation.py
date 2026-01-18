from django.db import models
from django.db.migrations.operations.base import Operation
from django.db.migrations.state import ModelState
from django.db.migrations.utils import field_references, resolve_relation
from django.db.models.options import normalize_together
from django.utils.functional import cached_property
from .fields import AddField, AlterField, FieldOperation, RemoveField, RenameField
class ModelOptionOperation(ModelOperation):

    def reduce(self, operation, app_label):
        if isinstance(operation, (self.__class__, DeleteModel)) and self.name_lower == operation.name_lower:
            return [operation]
        return super().reduce(operation, app_label)