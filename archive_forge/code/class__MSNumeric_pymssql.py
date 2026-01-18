import re
from .base import MSDialect
from .base import MSIdentifierPreparer
from ... import types as sqltypes
from ... import util
from ...engine import processors
class _MSNumeric_pymssql(sqltypes.Numeric):

    def result_processor(self, dialect, type_):
        if not self.asdecimal:
            return processors.to_float
        else:
            return sqltypes.Numeric.result_processor(self, dialect, type_)