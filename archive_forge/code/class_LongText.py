from oslo_serialization import jsonutils
from sqlalchemy.dialects import mysql
from sqlalchemy import types
class LongText(types.TypeDecorator):
    impl = types.Text
    cache_ok = True

    def load_dialect_impl(self, dialect):
        if dialect.name == 'mysql':
            return dialect.type_descriptor(mysql.LONGTEXT())
        else:
            return self.impl