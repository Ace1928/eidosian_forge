from pyrsistent._checked_types import CheckedType, _restore_pickle, InvariantException, store_invariants
from pyrsistent._field_common import (
from pyrsistent._pmap import PMap, pmap
class PRecord(PMap, CheckedType, metaclass=_PRecordMeta):
    """
    A PRecord is a PMap with a fixed set of specified fields. Records are declared as python classes inheriting
    from PRecord. Because it is a PMap it has full support for all Mapping methods such as iteration and element
    access using subscript notation.

    More documentation and examples of PRecord usage is available at https://github.com/tobgu/pyrsistent
    """

    def __new__(cls, **kwargs):
        if '_precord_size' in kwargs and '_precord_buckets' in kwargs:
            return super(PRecord, cls).__new__(cls, kwargs['_precord_size'], kwargs['_precord_buckets'])
        factory_fields = kwargs.pop('_factory_fields', None)
        ignore_extra = kwargs.pop('_ignore_extra', False)
        initial_values = kwargs
        if cls._precord_initial_values:
            initial_values = dict(((k, v() if callable(v) else v) for k, v in cls._precord_initial_values.items()))
            initial_values.update(kwargs)
        e = _PRecordEvolver(cls, pmap(pre_size=len(cls._precord_fields)), _factory_fields=factory_fields, _ignore_extra=ignore_extra)
        for k, v in initial_values.items():
            e[k] = v
        return e.persistent()

    def set(self, *args, **kwargs):
        """
        Set a field in the record. This set function differs slightly from that in the PMap
        class. First of all it accepts key-value pairs. Second it accepts multiple key-value
        pairs to perform one, atomic, update of multiple fields.
        """
        if args:
            return super(PRecord, self).set(args[0], args[1])
        return self.update(kwargs)

    def evolver(self):
        """
        Returns an evolver of this object.
        """
        return _PRecordEvolver(self.__class__, self)

    def __repr__(self):
        return '{0}({1})'.format(self.__class__.__name__, ', '.join(('{0}={1}'.format(k, repr(v)) for k, v in self.items())))

    @classmethod
    def create(cls, kwargs, _factory_fields=None, ignore_extra=False):
        """
        Factory method. Will create a new PRecord of the current type and assign the values
        specified in kwargs.

        :param ignore_extra: A boolean which when set to True will ignore any keys which appear in kwargs that are not
                             in the set of fields on the PRecord.
        """
        if isinstance(kwargs, cls):
            return kwargs
        if ignore_extra:
            kwargs = {k: kwargs[k] for k in cls._precord_fields if k in kwargs}
        return cls(_factory_fields=_factory_fields, _ignore_extra=ignore_extra, **kwargs)

    def __reduce__(self):
        return (_restore_pickle, (self.__class__, dict(self)))

    def serialize(self, format=None):
        """
        Serialize the current PRecord using custom serializer functions for fields where
        such have been supplied.
        """
        return dict(((k, serialize(self._precord_fields[k].serializer, format, v)) for k, v in self.items()))