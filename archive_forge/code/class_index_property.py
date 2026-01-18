from .. import inspect
from ..ext.hybrid import hybrid_property
from ..orm.attributes import flag_modified
class index_property(hybrid_property):
    """A property generator. The generated property describes an object
    attribute that corresponds to an :class:`_types.Indexable`
    column.

    .. seealso::

        :mod:`sqlalchemy.ext.indexable`

    """
    _NO_DEFAULT_ARGUMENT = object()

    def __init__(self, attr_name, index, default=_NO_DEFAULT_ARGUMENT, datatype=None, mutable=True, onebased=True):
        """Create a new :class:`.index_property`.

        :param attr_name:
            An attribute name of an `Indexable` typed column, or other
            attribute that returns an indexable structure.
        :param index:
            The index to be used for getting and setting this value.  This
            should be the Python-side index value for integers.
        :param default:
            A value which will be returned instead of `AttributeError`
            when there is not a value at given index.
        :param datatype: default datatype to use when the field is empty.
            By default, this is derived from the type of index used; a
            Python list for an integer index, or a Python dictionary for
            any other style of index.   For a list, the list will be
            initialized to a list of None values that is at least
            ``index`` elements long.
        :param mutable: if False, writes and deletes to the attribute will
            be disallowed.
        :param onebased: assume the SQL representation of this value is
            one-based; that is, the first index in SQL is 1, not zero.
        """
        if mutable:
            super().__init__(self.fget, self.fset, self.fdel, self.expr)
        else:
            super().__init__(self.fget, None, None, self.expr)
        self.attr_name = attr_name
        self.index = index
        self.default = default
        is_numeric = isinstance(index, int)
        onebased = is_numeric and onebased
        if datatype is not None:
            self.datatype = datatype
        elif is_numeric:
            self.datatype = lambda: [None for x in range(index + 1)]
        else:
            self.datatype = dict
        self.onebased = onebased

    def _fget_default(self, err=None):
        if self.default == self._NO_DEFAULT_ARGUMENT:
            raise AttributeError(self.attr_name) from err
        else:
            return self.default

    def fget(self, instance):
        attr_name = self.attr_name
        column_value = getattr(instance, attr_name)
        if column_value is None:
            return self._fget_default()
        try:
            value = column_value[self.index]
        except (KeyError, IndexError) as err:
            return self._fget_default(err)
        else:
            return value

    def fset(self, instance, value):
        attr_name = self.attr_name
        column_value = getattr(instance, attr_name, None)
        if column_value is None:
            column_value = self.datatype()
            setattr(instance, attr_name, column_value)
        column_value[self.index] = value
        setattr(instance, attr_name, column_value)
        if attr_name in inspect(instance).mapper.attrs:
            flag_modified(instance, attr_name)

    def fdel(self, instance):
        attr_name = self.attr_name
        column_value = getattr(instance, attr_name)
        if column_value is None:
            raise AttributeError(self.attr_name)
        try:
            del column_value[self.index]
        except KeyError as err:
            raise AttributeError(self.attr_name) from err
        else:
            setattr(instance, attr_name, column_value)
            flag_modified(instance, attr_name)

    def expr(self, model):
        column = getattr(model, self.attr_name)
        index = self.index
        if self.onebased:
            index += 1
        return column[index]