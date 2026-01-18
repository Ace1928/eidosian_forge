from django.db.backends.base.schema import BaseDatabaseSchemaEditor
from django.db.models import NOT_PROVIDED, F, UniqueConstraint
from django.db.models.constants import LOOKUP_SEP
def _column_default_sql(self, field):
    if not self.connection.mysql_is_mariadb and self._supports_limited_data_type_defaults and self._is_limited_data_type(field):
        return '(%s)'
    return super()._column_default_sql(field)