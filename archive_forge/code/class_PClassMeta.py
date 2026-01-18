from pyrsistent._checked_types import (InvariantException, CheckedType, _restore_pickle, store_invariants)
from pyrsistent._field_common import (
from pyrsistent._transformations import transform
class PClassMeta(type):

    def __new__(mcs, name, bases, dct):
        set_fields(dct, bases, name='_pclass_fields')
        store_invariants(dct, bases, '_pclass_invariants', '__invariant__')
        dct['__slots__'] = ('_pclass_frozen',) + tuple((key for key in dct['_pclass_fields']))
        if _is_pclass(bases):
            dct['__slots__'] += ('__weakref__',)
        return super(PClassMeta, mcs).__new__(mcs, name, bases, dct)