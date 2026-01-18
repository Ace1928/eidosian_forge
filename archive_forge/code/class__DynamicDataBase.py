from pyomo.core.base.set import Set
from pyomo.contrib.mpc.data.get_cuid import get_indexed_cuid
class _DynamicDataBase(object):
    """
    A base class for storing data associated with time-indexed variables.

    """

    def __init__(self, data, time_set=None, context=None):
        """
        Processes keys of the data dict.

        """
        self._orig_time_set = time_set
        self._data = {get_indexed_cuid(key, (self._orig_time_set,), context=context): val for key, val in data.items()}

    def __eq__(self, other):
        if isinstance(other, _DynamicDataBase):
            return self._data == other._data
        else:
            raise TypeError('%s and %s are not comparable' % (self.__class__, other.__class__))

    def get_data(self):
        """
        Return a dictionary mapping CUIDs to values

        """
        return self._data

    def get_cuid(self, key, context=None):
        """
        Get the time-indexed CUID corresponding to the provided key
        """
        return get_indexed_cuid(key, (self._orig_time_set,), context=context)

    def get_data_from_key(self, key, context=None):
        """
        Returns the value associated with the given key.

        """
        cuid = get_indexed_cuid(key, (self._orig_time_set,), context=context)
        return self._data[cuid]

    def contains_key(self, key, context=None):
        """
        Returns whether this object's dict contains the given key.

        """
        cuid = get_indexed_cuid(key, (self._orig_time_set,), context=context)
        return cuid in self._data

    def update_data(self, other, context=None):
        """
        Updates this object's data dict.

        """
        if isinstance(other, _DynamicDataBase):
            self._data.update(other.get_data())
        else:
            other = {get_indexed_cuid(key, (self._orig_time_set,), context=context): val for key, val in other.items()}
            self._data.update(other)

    def to_serializable(self):
        """
        Returns a json-serializable object.

        """
        raise NotImplementedError('to_serializable has not been implemented by %s' % self.__class__)

    def extract_variables(self, variables, context=None, copy_values=False):
        """
        Return a new object that only keeps data values for the variables
        specified.

        """
        if copy_values:
            raise NotImplementedError('extract_variables with copy_values=True has not been implemented by %s' % self.__class__)
        data = {}
        for var in variables:
            cuid = get_indexed_cuid(var, (self._orig_time_set,), context=context)
            data[cuid] = self._data[cuid]
        MyClass = self.__class__
        return MyClass(data, time_set=self._orig_time_set)