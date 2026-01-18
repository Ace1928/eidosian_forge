from django.db.backends.base.schema import BaseDatabaseSchemaEditor
from django.db.models import NOT_PROVIDED, F, UniqueConstraint
from django.db.models.constants import LOOKUP_SEP
def _field_db_check(self, field, field_db_params):
    if self.connection.mysql_is_mariadb and self.connection.mysql_version >= (10, 5, 2):
        return super()._field_db_check(field, field_db_params)
    return field_db_params['check']