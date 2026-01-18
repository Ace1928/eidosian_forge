from types import FunctionType, MethodType
import warnings
from .constants import (
from .ctrait import CTrait
from .trait_errors import TraitError
from .trait_base import (
from .trait_converters import (
from .trait_handler import TraitHandler
from .trait_type import (
from .trait_handlers import (
from .trait_factory import (
from .util.deprecated import deprecated
class _TraitMaker(object):
    type_map = {'event': TraitKind.event, 'constant': TraitKind.constant}

    def __init__(self, *value_type, **metadata):
        metadata.setdefault('type', 'trait')
        self.define(*value_type, **metadata)

    def define(self, *value_type, **metadata):
        """ Define the trait. """
        default_value_type = DefaultValue.unspecified
        default_value = handler = clone = None
        if len(value_type) > 0:
            default_value = value_type[0]
            value_type = value_type[1:]
            if len(value_type) == 0 and type(default_value) in SequenceTypes:
                default_value, value_type = (default_value[0], default_value)
            if len(value_type) == 0:
                default_value = try_trait_cast(default_value)
                if default_value in PythonTypes:
                    handler = TraitCoerceType(default_value)
                    default_value = DefaultValues.get(default_value)
                elif isinstance(default_value, CTrait):
                    clone = default_value
                    default_value_type, default_value = clone.default_value()
                    metadata['type'] = clone.type
                elif isinstance(default_value, TraitHandler):
                    handler = default_value
                    default_value = None
                else:
                    typeValue = type(default_value)
                    if typeValue in TypeTypes:
                        handler = TraitCastType(typeValue)
                    else:
                        metadata.setdefault('instance_handler', '_instance_changed_handler')
                        handler = TraitInstance(default_value)
                        if default_value is handler.aClass:
                            default_value = DefaultValues.get(default_value)
            else:
                enum = []
                other = []
                map = {}
                self.do_list(value_type, enum, map, other)
                if (len(enum) == 1 and enum[0] is None) and (len(other) == 1 and isinstance(other[0], TraitInstance)):
                    enum = []
                    other[0].allow_none()
                    metadata.setdefault('instance_handler', '_instance_changed_handler')
                if len(enum) > 0:
                    if len(map) + len(other) == 0 and default_value not in enum:
                        enum.insert(0, default_value)
                    other.append(TraitEnum(enum))
                if len(map) > 0:
                    other.append(TraitMap(map))
                if len(other) == 0:
                    handler = TraitHandler()
                elif len(other) == 1:
                    handler = other[0]
                    if isinstance(handler, CTrait):
                        clone, handler = (handler, None)
                        metadata['type'] = clone.type
                    elif isinstance(handler, TraitInstance):
                        metadata.setdefault('instance_handler', '_instance_changed_handler')
                        if default_value is None:
                            handler.allow_none()
                        elif isinstance(default_value, _InstanceArgs):
                            default_value_type = DefaultValue.callable_and_args
                            default_value = (handler.create_default_value, default_value.args, default_value.kw)
                        elif len(enum) == 0 and len(map) == 0:
                            aClass = handler.aClass
                            typeValue = type(default_value)
                            if typeValue is dict:
                                default_value_type = DefaultValue.callable_and_args
                                default_value = (aClass, (), default_value)
                            elif not isinstance(default_value, aClass):
                                if typeValue is not tuple:
                                    default_value = (default_value,)
                                default_value_type = DefaultValue.callable_and_args
                                default_value = (aClass, default_value, None)
                else:
                    for i, item in enumerate(other):
                        if isinstance(item, CTrait):
                            if item.type != 'trait':
                                raise TraitError('Cannot create a complex trait containing %s trait.' % add_article(item.type))
                            handler = item.handler
                            if handler is None:
                                break
                            other[i] = handler
                    else:
                        handler = TraitCompound(other)
        self.handler = handler
        self.clone = clone
        if default_value_type < 0:
            if isinstance(default_value, Default):
                default_value_type = DefaultValue.callable_and_args
                default_value = default_value.default_value
            else:
                if handler is None and clone is not None:
                    handler = clone.handler
                if handler is not None:
                    default_value_type = handler.default_value_type
                    if default_value_type < 0:
                        try:
                            default_value = handler.validate(None, '', default_value)
                        except:
                            pass
                if default_value_type < 0:
                    default_value_type = _infer_default_value_type(default_value)
        self.default_value_type = default_value_type
        self.default_value = default_value
        self.metadata = metadata.copy()

    def do_list(self, list, enum, map, other):
        """ Determine the correct TraitHandler for each item in a list. """
        for item in list:
            if item in PythonTypes:
                other.append(TraitCoerceType(item))
            else:
                item = try_trait_cast(item)
                typeItem = type(item)
                if typeItem in ConstantTypes:
                    enum.append(item)
                elif typeItem in SequenceTypes:
                    self.do_list(item, enum, map, other)
                elif typeItem is dict:
                    map.update(item)
                elif typeItem in CallableTypes:
                    other.append(TraitFunction(item))
                elif isinstance(item, TraitTypes):
                    other.append(item)
                else:
                    other.append(TraitInstance(item))

    def as_ctrait(self):
        """ Return a properly initialized 'CTrait' instance. """
        metadata = self.metadata
        trait = CTrait(self.type_map.get(metadata.get('type'), TraitKind.trait))
        clone = self.clone
        if clone is not None:
            trait.clone(clone)
            if clone.__dict__ is not None:
                trait.__dict__ = clone.__dict__.copy()
        trait.set_default_value(self.default_value_type, self.default_value)
        handler = self.handler
        if handler is not None:
            trait.handler = handler
            validate = getattr(handler, 'fast_validate', None)
            if validate is None:
                validate = handler.validate
            trait.set_validate(validate)
            post_setattr = getattr(handler, 'post_setattr', None)
            if post_setattr is not None:
                trait.post_setattr = post_setattr
                trait.is_mapped = handler.is_mapped
        rich_compare = metadata.get('rich_compare')
        if rich_compare is not None:
            warnings.warn("The 'rich_compare' metadata has been deprecated. Please use the 'comparison_mode' metadata instead. In a future release, rich_compare will have no effect.", DeprecationWarning, stacklevel=4)
            if rich_compare:
                trait.comparison_mode = ComparisonMode.equality
            else:
                trait.comparison_mode = ComparisonMode.identity
        comparison_mode = metadata.pop('comparison_mode', None)
        if comparison_mode is not None:
            trait.comparison_mode = comparison_mode
        if len(metadata) > 0:
            if trait.__dict__ is None:
                trait.__dict__ = metadata
            else:
                trait.__dict__.update(metadata)
        return trait