import re
from uuid import UUID as _python_UUID
from .base import MySQLCompiler
from .base import MySQLDialect
from .base import MySQLExecutionContext
from ... import sql
from ... import util
from ...sql import sqltypes
class _MariaDBUUID(sqltypes.UUID[sqltypes._UUID_RETURN]):

    def result_processor(self, dialect, coltype):
        if self.as_uuid:

            def process(value):
                if value is not None:
                    if hasattr(value, 'decode'):
                        value = value.decode('ascii')
                    value = _python_UUID(value)
                return value
            return process
        else:

            def process(value):
                if value is not None:
                    if hasattr(value, 'decode'):
                        value = value.decode('ascii')
                    value = str(_python_UUID(value))
                return value
            return process