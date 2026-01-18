from abc import ABCMeta
from typing import Any, Dict, Optional, Pattern, Tuple, Type
import re
class AtomicTypeMeta(ABCMeta):
    """
    Metaclass for creating XSD atomic types. The created classes
    are decorated with missing attributes and methods. When a name
    attribute is provided the class is registered into a global map
    of XSD atomic types and also the expanded name is added.
    """
    xsd_version: str
    pattern: Pattern[str]
    name: Optional[str] = None

    def __new__(mcs, class_name: str, bases: Tuple[Type[Any], ...], dict_: Dict[str, Any]) -> 'AtomicTypeMeta':
        try:
            name = dict_['name']
        except KeyError:
            name = dict_['name'] = None
        if name is not None and (not isinstance(name, str)):
            raise TypeError("attribute 'name' must be a string or None")
        dict_['is_valid'] = classmethod(mcs.is_valid)
        dict_['invalid_type'] = classmethod(mcs.invalid_type)
        dict_['invalid_value'] = classmethod(mcs.invalid_value)
        cls = super(AtomicTypeMeta, mcs).__new__(mcs, class_name, bases, dict_)
        if not hasattr(cls, 'xsd_version'):
            cls.xsd_version = '1.0'
        if not hasattr(cls, 'pattern'):
            cls.pattern = re.compile('^$')
        if name:
            expanded_name = '{%s}%s' % (XSD_NAMESPACE, name)
            if cls.xsd_version == '1.0':
                xsd10_atomic_types[name] = xsd10_atomic_types[expanded_name] = cls
            else:
                xsd11_atomic_types[name] = xsd11_atomic_types[expanded_name] = cls
        return cls

    def validate(cls: Type[Any], value: object) -> None:
        if isinstance(value, cls):
            return
        elif isinstance(value, str):
            if cls.pattern.match(value) is None:
                raise cls.invalid_value(value)
        else:
            raise cls.invalid_type(value)

    def is_valid(cls: Type[Any], value: object) -> bool:
        try:
            cls.validate(value)
        except (TypeError, ValueError):
            return False
        else:
            return True

    def invalid_type(cls: Type[Any], value: object) -> TypeError:
        if cls.name:
            return TypeError('invalid type {!r} for xs:{}'.format(type(value), cls.name))
        return TypeError('invalid type {!r} for {!r}'.format(type(value), cls))

    def invalid_value(cls: Type[Any], value: object) -> ValueError:
        if cls.name:
            return ValueError('invalid value {!r} for xs:{}'.format(value, cls.name))
        return ValueError('invalid value {!r} for {!r}'.format(value, cls))