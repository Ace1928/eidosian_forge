import re
from .types import _StringType
from ... import exc
from ... import sql
from ... import util
from ...sql import sqltypes
class ENUM(sqltypes.NativeForEmulated, sqltypes.Enum, _StringType):
    """MySQL ENUM type."""
    __visit_name__ = 'ENUM'
    native_enum = True

    def __init__(self, *enums, **kw):
        """Construct an ENUM.

        E.g.::

          Column('myenum', ENUM("foo", "bar", "baz"))

        :param enums: The range of valid values for this ENUM.  Values in
          enums are not quoted, they will be escaped and surrounded by single
          quotes when generating the schema.  This object may also be a
          PEP-435-compliant enumerated type.

          .. versionadded: 1.1 added support for PEP-435-compliant enumerated
             types.

        :param strict: This flag has no effect.

         .. versionchanged:: The MySQL ENUM type as well as the base Enum
            type now validates all Python data values.

        :param charset: Optional, a column-level character set for this string
          value.  Takes precedence to 'ascii' or 'unicode' short-hand.

        :param collation: Optional, a column-level collation for this string
          value.  Takes precedence to 'binary' short-hand.

        :param ascii: Defaults to False: short-hand for the ``latin1``
          character set, generates ASCII in schema.

        :param unicode: Defaults to False: short-hand for the ``ucs2``
          character set, generates UNICODE in schema.

        :param binary: Defaults to False: short-hand, pick the binary
          collation type that matches the column's character set.  Generates
          BINARY in schema.  This does not affect the type of data stored,
          only the collation of character data.

        """
        kw.pop('strict', None)
        self._enum_init(enums, kw)
        _StringType.__init__(self, length=self.length, **kw)

    @classmethod
    def adapt_emulated_to_native(cls, impl, **kw):
        """Produce a MySQL native :class:`.mysql.ENUM` from plain
        :class:`.Enum`.

        """
        kw.setdefault('validate_strings', impl.validate_strings)
        kw.setdefault('values_callable', impl.values_callable)
        kw.setdefault('omit_aliases', impl._omit_aliases)
        return cls(**kw)

    def _object_value_for_elem(self, elem):
        if elem == '':
            return elem
        else:
            return super()._object_value_for_elem(elem)

    def __repr__(self):
        return util.generic_repr(self, to_inspect=[ENUM, _StringType, sqltypes.Enum])