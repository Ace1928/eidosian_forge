from django.db.backends.postgresql.schema import DatabaseSchemaEditor
from django.db.models.expressions import Col, Func
def _field_should_be_indexed(self, model, field):
    if getattr(field, 'spatial_index', False):
        return True
    return super()._field_should_be_indexed(model, field)