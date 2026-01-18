from django.db.backends.base.schema import BaseDatabaseSchemaEditor
from django.db.models import NOT_PROVIDED, F, UniqueConstraint
from django.db.models.constants import LOOKUP_SEP
def _create_missing_fk_index(self, model, *, fields, expressions=None):
    """
        MySQL can remove an implicit FK index on a field when that field is
        covered by another index like a unique_together. "covered" here means
        that the more complex index has the FK field as its first field (see
        https://bugs.mysql.com/bug.php?id=37910).

        Manually create an implicit FK index to make it possible to remove the
        composed index.
        """
    first_field_name = None
    if fields:
        first_field_name = fields[0]
    elif expressions and self.connection.features.supports_expression_indexes and isinstance(expressions[0], F) and (LOOKUP_SEP not in expressions[0].name):
        first_field_name = expressions[0].name
    if not first_field_name:
        return
    first_field = model._meta.get_field(first_field_name)
    if first_field.get_internal_type() == 'ForeignKey':
        column = self.connection.introspection.identifier_converter(first_field.column)
        with self.connection.cursor() as cursor:
            constraint_names = [name for name, infodict in self.connection.introspection.get_constraints(cursor, model._meta.db_table).items() if infodict['index'] and infodict['columns'][0] == column]
        if len(constraint_names) == 1:
            self.execute(self._create_index_sql(model, fields=[first_field], suffix=''))