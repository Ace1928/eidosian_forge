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
class NumericSqlVariant(TypeDecorator):
    """This type casts sql_variant columns in the identity_columns view
    to numeric. This is required because:

    * pyodbc does not support sql_variant
    * pymssql under python 2 return the byte representation of the number,
      int 1 is returned as "\\x01\\x00\\x00\\x00". On python 3 it returns the
      correct value as string.
    """
    impl = Unicode
    cache_ok = True

    def column_expression(self, colexpr):
        return cast(colexpr, Numeric(38, 0))