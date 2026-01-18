import datetime
import math
import typing as t
from wandb.util import (
class TypedDictType(Type):
    """Represents a dictionary object where each key can have a type."""
    name = 'typedDict'
    legacy_names = ['dictionary']
    types: t.ClassVar[t.List[type]] = [dict]

    def __init__(self, type_map: t.Optional[t.Dict[str, ConvertableToType]]=None):
        if type_map is None:
            type_map = {}
        self.params.update({'type_map': {key: TypeRegistry.type_from_dtype(type_map[key]) for key in type_map}})

    @classmethod
    def from_obj(cls, py_obj: t.Optional[t.Any]=None) -> 'TypedDictType':
        if not isinstance(py_obj, dict):
            TypeError('TypedDictType.from_obj expects a dictionary')
        assert isinstance(py_obj, dict)
        return cls({key: TypeRegistry.type_of(py_obj[key]) for key in py_obj})

    def assign_type(self, wb_type: 'Type') -> t.Union['TypedDictType', InvalidType]:
        if isinstance(wb_type, TypedDictType) and len(set(wb_type.params['type_map'].keys()) - set(self.params['type_map'].keys())) == 0:
            type_map = {}
            for key in self.params['type_map']:
                type_map[key] = self.params['type_map'][key].assign_type(wb_type.params['type_map'].get(key, UnknownType()))
                if isinstance(type_map[key], InvalidType):
                    return InvalidType()
            return TypedDictType(type_map)
        return InvalidType()

    def assign(self, py_obj: t.Optional[t.Any]=None) -> t.Union['TypedDictType', InvalidType]:
        if isinstance(py_obj, dict) and len(set(py_obj.keys()) - set(self.params['type_map'].keys())) == 0:
            type_map = {}
            for key in self.params['type_map']:
                type_map[key] = self.params['type_map'][key].assign(py_obj.get(key, None))
                if isinstance(type_map[key], InvalidType):
                    return InvalidType()
            return TypedDictType(type_map)
        return InvalidType()

    def explain(self, other: t.Any, depth=0) -> str:
        exp = super().explain(other, depth)
        gap = ''.join(['\t'] * depth)
        if isinstance(other, dict):
            extra_keys = set(other.keys()) - set(self.params['type_map'].keys())
            if len(extra_keys) > 0:
                exp += '\n{}Found extra keys: {}'.format(gap, ','.join(list(extra_keys)))
            for key in self.params['type_map']:
                val = other.get(key, None)
                if isinstance(self.params['type_map'][key].assign(val), InvalidType):
                    exp += "\n{}Key '{}':\n{}".format(gap, key, self.params['type_map'][key].explain(val, depth=depth + 1))
        return exp

    def __repr__(self):
        return '{}'.format(self.params['type_map'])