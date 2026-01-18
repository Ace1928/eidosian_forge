import re
from .types import _StringType
from ... import exc
from ... import sql
from ... import util
from ...sql import sqltypes
class SET(_StringType):
    """MySQL SET type."""
    __visit_name__ = 'SET'

    def __init__(self, *values, **kw):
        """Construct a SET.

        E.g.::

          Column('myset', SET("foo", "bar", "baz"))


        The list of potential values is required in the case that this
        set will be used to generate DDL for a table, or if the
        :paramref:`.SET.retrieve_as_bitwise` flag is set to True.

        :param values: The range of valid values for this SET. The values
          are not quoted, they will be escaped and surrounded by single
          quotes when generating the schema.

        :param convert_unicode: Same flag as that of
         :paramref:`.String.convert_unicode`.

        :param collation: same as that of :paramref:`.String.collation`

        :param charset: same as that of :paramref:`.VARCHAR.charset`.

        :param ascii: same as that of :paramref:`.VARCHAR.ascii`.

        :param unicode: same as that of :paramref:`.VARCHAR.unicode`.

        :param binary: same as that of :paramref:`.VARCHAR.binary`.

        :param retrieve_as_bitwise: if True, the data for the set type will be
          persisted and selected using an integer value, where a set is coerced
          into a bitwise mask for persistence.  MySQL allows this mode which
          has the advantage of being able to store values unambiguously,
          such as the blank string ``''``.   The datatype will appear
          as the expression ``col + 0`` in a SELECT statement, so that the
          value is coerced into an integer value in result sets.
          This flag is required if one wishes
          to persist a set that can store the blank string ``''`` as a value.

          .. warning::

            When using :paramref:`.mysql.SET.retrieve_as_bitwise`, it is
            essential that the list of set values is expressed in the
            **exact same order** as exists on the MySQL database.

        """
        self.retrieve_as_bitwise = kw.pop('retrieve_as_bitwise', False)
        self.values = tuple(values)
        if not self.retrieve_as_bitwise and '' in values:
            raise exc.ArgumentError("Can't use the blank value '' in a SET without setting retrieve_as_bitwise=True")
        if self.retrieve_as_bitwise:
            self._bitmap = {value: 2 ** idx for idx, value in enumerate(self.values)}
            self._bitmap.update(((2 ** idx, value) for idx, value in enumerate(self.values)))
        length = max([len(v) for v in values] + [0])
        kw.setdefault('length', length)
        super().__init__(**kw)

    def column_expression(self, colexpr):
        if self.retrieve_as_bitwise:
            return sql.type_coerce(sql.type_coerce(colexpr, sqltypes.Integer) + 0, self)
        else:
            return colexpr

    def result_processor(self, dialect, coltype):
        if self.retrieve_as_bitwise:

            def process(value):
                if value is not None:
                    value = int(value)
                    return set(util.map_bits(self._bitmap.__getitem__, value))
                else:
                    return None
        else:
            super_convert = super().result_processor(dialect, coltype)

            def process(value):
                if isinstance(value, str):
                    if super_convert:
                        value = super_convert(value)
                    return set(re.findall('[^,]+', value))
                else:
                    if value is not None:
                        value.discard('')
                    return value
        return process

    def bind_processor(self, dialect):
        super_convert = super().bind_processor(dialect)
        if self.retrieve_as_bitwise:

            def process(value):
                if value is None:
                    return None
                elif isinstance(value, (int, str)):
                    if super_convert:
                        return super_convert(value)
                    else:
                        return value
                else:
                    int_value = 0
                    for v in value:
                        int_value |= self._bitmap[v]
                    return int_value
        else:

            def process(value):
                if value is not None and (not isinstance(value, (int, str))):
                    value = ','.join(value)
                if super_convert:
                    return super_convert(value)
                else:
                    return value
        return process

    def adapt(self, impltype, **kw):
        kw['retrieve_as_bitwise'] = self.retrieve_as_bitwise
        return util.constructor_copy(self, impltype, *self.values, **kw)

    def __repr__(self):
        return util.generic_repr(self, to_inspect=[SET, _StringType], additional_kw=[('retrieve_as_bitwise', False)])