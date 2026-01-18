from django.db.backends.base.schema import BaseDatabaseSchemaEditor
from django.db.models import NOT_PROVIDED, F, UniqueConstraint
from django.db.models.constants import LOOKUP_SEP
def _delete_composed_index(self, model, fields, *args):
    self._create_missing_fk_index(model, fields=fields)
    return super()._delete_composed_index(model, fields, *args)