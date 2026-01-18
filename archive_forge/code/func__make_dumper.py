from __future__ import annotations
import dataclasses
import enum
import functools
from typing import IO, TYPE_CHECKING, Any, Optional, Set, Type, TypeVar, Union
from typing_extensions import get_args, get_origin
from .. import _fields, _resolver
def _make_dumper(instance: Any) -> Type[yaml.Dumper]:
    import yaml

    class DataclassDumper(yaml.Dumper):

        def ignore_aliases(self, data):
            return super().ignore_aliases(data) or data is _fields.MISSING_PROP
    contained_types = list(_get_contained_special_types_from_type(type(instance)))
    contained_type_names = list(map(lambda cls: cls.__name__, contained_types))
    assert len(set(contained_type_names)) == len(contained_type_names), f'Contained dataclass/enum names must all be unique, but got {contained_type_names}'

    def make_representer(name: str):

        def representer(dumper: DataclassDumper, data: Any) -> yaml.Node:
            if dataclasses.is_dataclass(data):
                return dumper.represent_mapping(tag=DATACLASS_YAML_TAG_PREFIX + name, mapping={field.name: getattr(data, field.name) for field in dataclasses.fields(data) if field.init})
            elif isinstance(data, enum.Enum):
                return dumper.represent_scalar(tag=ENUM_YAML_TAG_PREFIX + name, value=data.name)
            assert False
        return representer
    for typ, name in zip(contained_types, contained_type_names):
        DataclassDumper.add_representer(typ, make_representer(name))
    DataclassDumper.add_representer(type(_fields.MISSING_PROP), lambda dumper, data: dumper.represent_scalar(tag=MISSING_YAML_TAG_PREFIX, value=''))
    return DataclassDumper