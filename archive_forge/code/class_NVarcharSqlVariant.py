from ... import cast
from ... import Column
from ... import MetaData
from ... import Table
from ...ext.compiler import compiles
from ...sql import expression
from ...types import Boolean
from ...types import Integer
from ...types import Numeric
from ...types import NVARCHAR
from ...types import String
from ...types import TypeDecorator
from ...types import Unicode
class NVarcharSqlVariant(TypeDecorator):
    """This type casts sql_variant columns in the extended_properties view
    to nvarchar. This is required because pyodbc does not support sql_variant
    """
    impl = Unicode
    cache_ok = True

    def column_expression(self, colexpr):
        return cast(colexpr, NVARCHAR)