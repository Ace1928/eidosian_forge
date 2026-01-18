import datetime
import math
import typing as t
from wandb.util import (
class ConstType(Type):
    """A constant value (currently only primitives supported)."""
    name = 'const'
    types: t.ClassVar[t.List[type]] = []

    def __init__(self, val: t.Optional[t.Any]=None, is_set: t.Optional[bool]=False):
        if val.__class__ not in [str, int, float, bool, set, list, None.__class__]:
            TypeError(f'ConstType only supports str, int, float, bool, set, list, and None types. Found {val}')
        if is_set or isinstance(val, set):
            is_set = True
            assert isinstance(val, set) or isinstance(val, list)
            val = set(val)
        self.params.update({'val': val, 'is_set': is_set})

    def assign(self, py_obj: t.Optional[t.Any]=None) -> 'Type':
        return self.assign_type(ConstType(py_obj))

    @classmethod
    def from_obj(cls, py_obj: t.Optional[t.Any]=None) -> 'ConstType':
        return cls(py_obj)

    def __repr__(self):
        return str(self.params['val'])