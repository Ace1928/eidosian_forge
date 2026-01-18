from typing import Any, Dict, Iterable, List, Optional, Union
import pyarrow as pa
from triad import Schema, assert_or_throw, to_uuid
from triad.utils.pyarrow import _type_to_expression, to_pa_datatype
class ColumnExpr:
    """Fugue column expression class. It is inspired from
    :class:`spark:pyspark.sql.Column` and it is working in progress.

    .. admonition:: New Since
            :class: hint

            **0.6.0**

    .. caution::

        This is a base class of different column classes, and users are not supposed
        to construct this class directly. Use :func:`~.col` and :func:`~.lit` instead.
    """

    def __init__(self):
        self._as_name = ''
        self._as_type: Optional[pa.DataType] = None

    @property
    def name(self) -> str:
        """The original name of this column, default empty

        :return: the name

        .. admonition:: Examples

            .. code-block:: python

                assert "a" == col("a").name
                assert "b" == col("a").alias("b").name
                assert "" == lit(1).name
                assert "" == (col("a") * 2).name
        """
        return ''

    @property
    def as_name(self) -> str:
        """The name assigned by :meth:`~.alias`

        :return: the alias

        .. admonition:: Examples

            .. code-block:: python

                assert "" == col("a").as_name
                assert "b" == col("a").alias("b").as_name
                assert "x" == (col("a") * 2).alias("x").as_name
        """
        return self._as_name

    @property
    def as_type(self) -> Optional[pa.DataType]:
        """The type assigned by :meth:`~.cast`

        :return: the pyarrow datatype if :meth:`~.cast` was called
          otherwise None

        .. admonition:: Examples

            .. code-block:: python

                import pyarrow as pa

                assert col("a").as_type is None
                assert pa.int64() == col("a").cast(int).as_type
                assert pa.string() == (col("a") * 2).cast(str).as_type
        """
        return self._as_type

    @property
    def output_name(self) -> str:
        """The name assigned by :meth:`~.alias`, but if empty then
        return the original column name

        :return: the alias or the original column name

        .. admonition:: Examples

            .. code-block:: python

                assert "a" == col("a").output_name
                assert "b" == col("a").alias("b").output_name
                assert "x" == (col("a") * 2).alias("x").output_name
        """
        return self.as_name if self.as_name != '' else self.name

    def alias(self, as_name: str) -> 'ColumnExpr':
        """Assign or remove alias of a column. To remove, set ``as_name`` to empty

        :return: a new column with the alias value

        .. admonition:: Examples

            .. code-block:: python

                assert "b" == col("a").alias("b").as_name
                assert "x" == (col("a") * 2).alias("x").as_name
                assert "" == col("a").alias("b").alias("").as_name
        """
        raise NotImplementedError

    def infer_alias(self) -> 'ColumnExpr':
        """Infer alias of a column. If the column's :meth:`~.output_name` is not empty
        then it returns itself without change. Otherwise it tries to infer alias from
        the underlying columns.

        :return: a column instance with inferred alias

        .. caution::

            Users should not use it directly.

        .. admonition:: Examples

            .. code-block:: python

                import fugue.column.functions as f

                assert "a" == col("a").infer_alias().output_name
                assert "" == (col("a") * 2).infer_alias().output_name
                assert "a" == col("a").is_null().infer_alias().output_name
                assert "a" == f.max(col("a").is_null()).infer_alias().output_name
        """
        return self

    def cast(self, data_type: Any) -> 'ColumnExpr':
        """Cast the column to a new data type

        :param data_type: It can be string expressions, python primitive types,
          python `datetime.datetime` and pyarrow types.
          For details read |FugueDataTypes|
        :return: a new column instance with the assigned data type

        .. caution::

            Currently, casting to struct or list type has undefined behavior.

        .. admonition:: Examples

            .. code-block:: python

                import pyarrow as pa

                assert pa.int64() == col("a").cast(int).as_type
                assert pa.string() == col("a").cast(str).as_type
                assert pa.float64() == col("a").cast(float).as_type
                assert pa._bool() == col("a").cast(bool).as_type

                # string follows the type expression of Triad Schema
                assert pa.int32() == col("a").cast("int").as_type
                assert pa.int32() == col("a").cast("int32").as_type

                assert pa.int32() == col("a").cast(pa.int32()).as_type
        """
        raise NotImplementedError

    def infer_type(self, schema: Schema) -> Optional[pa.DataType]:
        """Infer data type of this column given the input schema

        :param schema: the schema instance to infer from
        :return: a pyarrow datatype or None if failed to infer

        .. caution::

            Users should not use it directly.

        .. admonition:: Examples

            .. code-block:: python

                import pyarrow as pa
                from triad import Schema
                import fugue.column.functions as f

                schema = Schema("a:int,b:str")

                assert pa.int32() == col("a").infer_schema(schema)
                assert pa.int32() == (-col("a")).infer_schema(schema)
                # due to overflow risk, can't infer certain operations
                assert (col("a")+1).infer_schema(schema) is None
                assert (col("a")+col("a")).infer_schema(schema) is None
                assert pa.int32() == f.max(col("a")).infer_schema(schema)
                assert pa.int32() == f.min(col("a")).infer_schema(schema)
                assert f.sum(col("a")).infer_schema(schema) is None
        """
        return self.as_type

    def __str__(self) -> str:
        """String expression of the column, this is only used for debug purpose.
        It is not SQL expression.

        :return: the string expression
        """
        res = self.body_str
        if self.as_type is not None:
            res = f'CAST({res} AS {_type_to_expression(self.as_type)})'
        if self.as_name != '':
            res = res + ' AS ' + self.as_name
        return res

    @property
    def body_str(self) -> str:
        """The string expression of this column without cast type and alias.
        This is only used for debug purpose. It is not SQL expression.

        :return: the string expression
        """
        raise NotImplementedError

    def is_null(self) -> 'ColumnExpr':
        """Same as SQL ``<col> IS NULL``.

        :return: a new column with the boolean values
        """
        return _UnaryOpExpr('IS_NULL', self)

    def not_null(self) -> 'ColumnExpr':
        """Same as SQL ``<col> IS NOT NULL``.

        :return: a new column with the boolean values
        """
        return _UnaryOpExpr('NOT_NULL', self)

    def __neg__(self) -> 'ColumnExpr':
        """The negative value of the current column

        :return: a new column with the negative value
        """
        return _InvertOpExpr('-', self)

    def __pos__(self) -> 'ColumnExpr':
        """The original value of the current column

        :return: the column itself
        """
        return self

    def __invert__(self) -> 'ColumnExpr':
        """Same as SQL ``NOT <col>``

        :return: a new column with the boolean values
        """
        return _NotOpExpr('~', self)

    def __add__(self, other: Any) -> 'ColumnExpr':
        """Add with another column

        :param other: the other column, if it is not a
          :class:`~.ColumnExpr`, then the value will be converted to
          a literal (`lit(other)`)
        :return: a new column with the result
        """
        return _BinaryOpExpr('+', self, other)

    def __radd__(self, other: Any) -> 'ColumnExpr':
        """Add with another column

        :param other: the other column, if it is not a
          :class:`~.ColumnExpr`, then the value will be converted to
          a literal (`lit(other)`)
        :return: a new column with the result
        """
        return _BinaryOpExpr('+', other, self)

    def __sub__(self, other: Any) -> 'ColumnExpr':
        """Subtract another column from this column

        :param other: the other column, if it is not a
          :class:`~.ColumnExpr`, then the value will be converted to
          a literal (`lit(other)`)
        :return: a new column with the result
        """
        return _BinaryOpExpr('-', self, other)

    def __rsub__(self, other: Any) -> 'ColumnExpr':
        """Subtract this column from the other column

        :param other: the other column, if it is not a
          :class:`~.ColumnExpr`, then the value will be converted to
          a literal (`lit(other)`)
        :return: a new column with the result
        """
        return _BinaryOpExpr('-', other, self)

    def __mul__(self, other: Any) -> 'ColumnExpr':
        """Multiply with another column

        :param other: the other column, if it is not a
          :class:`~.ColumnExpr`, then the value will be converted to
          a literal (`lit(other)`)
        :return: a new column with the result
        """
        return _BinaryOpExpr('*', self, other)

    def __rmul__(self, other: Any) -> 'ColumnExpr':
        """Multiply with another column

        :param other: the other column, if it is not a
          :class:`~.ColumnExpr`, then the value will be converted to
          a literal (`lit(other)`)
        :return: a new column with the result
        """
        return _BinaryOpExpr('*', other, self)

    def __truediv__(self, other: Any) -> 'ColumnExpr':
        """Divide this column by the other column

        :param other: the other column, if it is not a
          :class:`~.ColumnExpr`, then the value will be converted to
          a literal (`lit(other)`)
        :return: a new column with the result
        """
        return _BinaryOpExpr('/', self, other)

    def __rtruediv__(self, other: Any) -> 'ColumnExpr':
        """Divide the other column by this column

        :param other: the other column, if it is not a
          :class:`~.ColumnExpr`, then the value will be converted to
          a literal (`lit(other)`)
        :return: a new column with the result
        """
        return _BinaryOpExpr('/', other, self)

    def __and__(self, other: Any) -> 'ColumnExpr':
        """``AND`` value of the two columns

        :param other: the other column, if it is not a
          :class:`~.ColumnExpr`, then the value will be converted to
          a literal (`lit(other)`)
        :return: a new column with the result
        """
        return _BoolBinaryOpExpr('&', self, other)

    def __rand__(self, other: Any) -> 'ColumnExpr':
        """``AND`` value of the two columns

        :param other: the other column, if it is not a
          :class:`~.ColumnExpr`, then the value will be converted to
          a literal (`lit(other)`)
        :return: a new column with the result
        """
        return _BoolBinaryOpExpr('&', other, self)

    def __or__(self, other: Any) -> 'ColumnExpr':
        """``OR`` value of the two columns

        :param other: the other column, if it is not a
          :class:`~.ColumnExpr`, then the value will be converted to
          a literal (`lit(other)`)
        :return: a new column with the result
        """
        return _BoolBinaryOpExpr('|', self, other)

    def __ror__(self, other: Any) -> 'ColumnExpr':
        """``OR`` value of the two columns

        :param other: the other column, if it is not a
          :class:`~.ColumnExpr`, then the value will be converted to
          a literal (`lit(other)`)
        :return: a new column with the result
        """
        return _BoolBinaryOpExpr('|', other, self)

    def __lt__(self, other: Any) -> 'ColumnExpr':
        """Whether this column is less than the other column

        :param other: the other column, if it is not a
          :class:`~.ColumnExpr`, then the value will be converted to
          a literal (`lit(other)`)
        :return: a new column with the boolean result
        """
        return _BoolBinaryOpExpr('<', self, other)

    def __gt__(self, other: Any) -> 'ColumnExpr':
        """Whether this column is greater than the other column

        :param other: the other column, if it is not a
          :class:`~.ColumnExpr`, then the value will be converted to
          a literal (`lit(other)`)
        :return: a new column with the boolean result
        """
        return _BoolBinaryOpExpr('>', self, other)

    def __le__(self, other: Any) -> 'ColumnExpr':
        """Whether this column is less or equal to the other column

        :param other: the other column, if it is not a
          :class:`~.ColumnExpr`, then the value will be converted to
          a literal (`lit(other)`)
        :return: a new column with the boolean result
        """
        return _BoolBinaryOpExpr('<=', self, other)

    def __ge__(self, other: Any) -> 'ColumnExpr':
        """Whether this column is greater or equal to the other column

        :param other: the other column, if it is not a
          :class:`~.ColumnExpr`, then the value will be converted to
          a literal (`lit(other)`)
        :return: a new column with the boolean result
        """
        return _BoolBinaryOpExpr('>=', self, other)

    def __eq__(self, other: Any) -> 'ColumnExpr':
        """Whether this column equals the other column

        :param other: the other column, if it is not a
          :class:`~.ColumnExpr`, then the value will be converted to
          a literal (`lit(other)`)
        :return: a new column with the boolean result
        """
        return _BoolBinaryOpExpr('==', self, other)

    def __ne__(self, other: Any) -> 'ColumnExpr':
        """Whether this column does not equal the other column

        :param other: the other column, if it is not a
          :class:`~.ColumnExpr`, then the value will be converted to
          a literal (`lit(other)`)
        :return: a new column with the boolean result
        """
        return _BoolBinaryOpExpr('!=', self, other)

    def __uuid__(self) -> str:
        """The unique id of this instance

        :return: the unique id
        """
        return to_uuid(str(type(self)), self.as_name, self.as_type, *self._uuid_keys())

    def _uuid_keys(self) -> List[Any]:
        raise NotImplementedError