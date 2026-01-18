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
class MetaHasProps(type):
    """ Specialize the construction of |HasProps| classes.

    This class is a `metaclass`_ for |HasProps| that is responsible for
    creating and adding the ``PropertyDescriptor`` instances that delegate
    validation and serialization to |Property| attributes.

    .. _metaclass: https://docs.python.org/3/reference/datamodel.html#metaclasses

    """
    __properties__: dict[str, Property[Any]]
    __overridden_defaults__: dict[str, Any]
    __themed_values__: dict[str, Any]

    def __new__(cls, class_name: str, bases: tuple[type, ...], class_dict: dict[str, Any]):
        """

        """
        overridden_defaults = _overridden_defaults(class_dict)
        generators = _generators(class_dict)
        properties = {}
        for name, generator in generators.items():
            descriptors = generator.make_descriptors(name)
            for descriptor in descriptors:
                name = descriptor.name
                if name in class_dict:
                    raise RuntimeError(f'Two property generators both created {class_name}.{name}')
                class_dict[name] = descriptor
                properties[name] = descriptor.property
        class_dict['__properties__'] = properties
        class_dict['__overridden_defaults__'] = overridden_defaults
        return super().__new__(cls, class_name, bases, class_dict)

    def __init__(cls, class_name: str, bases: tuple[type, ...], _) -> None:
        if class_name == 'HasProps':
            return
        base_properties: dict[str, Any] = {}
        for base in (x for x in bases if issubclass(x, HasProps)):
            base_properties.update(base.properties(_with_props=True))
        own_properties = {k: v for k, v in cls.__dict__.items() if isinstance(v, PropertyDescriptor)}
        redeclared = own_properties.keys() & base_properties.keys()
        if redeclared:
            warn(f'Properties {redeclared!r} in class {cls.__name__} were previously declared on a parent class. It never makes sense to do this. Redundant properties should be deleted here, or on the parent class. Override() can be used to change a default value of a base class property.', RuntimeWarning)
        unused_overrides = cls.__overridden_defaults__.keys() - cls.properties(_with_props=True).keys()
        if unused_overrides:
            warn(f'Overrides of {unused_overrides} in class {cls.__name__} does not override anything.', RuntimeWarning)

    @property
    def model_class_reverse_map(cls) -> dict[str, type[HasProps]]:
        return _default_resolver.known_models