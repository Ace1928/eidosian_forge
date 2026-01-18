from __future__ import annotations
import logging # isort:skip
import difflib
from typing import (
from weakref import WeakSet
from ..settings import settings
from ..util.strings import append_docstring, nice_join
from ..util.warnings import warn
from .property.descriptor_factory import PropertyDescriptorFactory
from .property.descriptors import PropertyDescriptor, UnsetValueError
from .property.override import Override
from .property.singletons import Intrinsic, Undefined
from .property.wrappers import PropertyValueContainer
from .serialization import (
from .types import ID
def _HasProps_to_serializable(cls: type[HasProps], serializer: Serializer) -> Ref | ModelDef:
    from ..model import DataModel, Model
    ref = Ref(id=ID(cls.__qualified_model__))
    serializer.add_ref(cls, ref)
    if not is_DataModel(cls):
        return ref
    bases: list[type[HasProps]] = [base for base in cls.__bases__ if issubclass(base, Model) and base != DataModel]
    if len(bases) == 0:
        extends = None
    elif len(bases) == 1:
        [base] = bases
        extends = serializer.encode(base)
    else:
        serializer.error('multiple bases are not supported')
    properties: list[PropertyDef] = []
    overrides: list[OverrideDef] = []
    for prop_name in cls.__properties__:
        descriptor = cls.lookup(prop_name)
        kind = 'Any'
        default = descriptor.property._default
        if default is Undefined:
            prop_def = PropertyDef(name=prop_name, kind=kind)
        else:
            if descriptor.is_unstable(default):
                default = default()
            prop_def = PropertyDef(name=prop_name, kind=kind, default=serializer.encode(default))
        properties.append(prop_def)
    for prop_name, default in getattr(cls, '__overridden_defaults__', {}).items():
        overrides.append(OverrideDef(name=prop_name, default=serializer.encode(default)))
    modeldef = ModelDef(type='model', name=cls.__qualified_model__)
    if extends is not None:
        modeldef['extends'] = extends
    if properties:
        modeldef['properties'] = properties
    if overrides:
        modeldef['overrides'] = overrides
    return modeldef