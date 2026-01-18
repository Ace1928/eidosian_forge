from django.db.backends.base.schema import BaseDatabaseSchemaEditor
from django.db.models import NOT_PROVIDED, F, UniqueConstraint
from django.db.models.constants import LOOKUP_SEP
def _is_limited_data_type(self, field):
    db_type = field.db_type(self.connection)
    return db_type is not None and db_type.lower() in self.connection._limited_data_types