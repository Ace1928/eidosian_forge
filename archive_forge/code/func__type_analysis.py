import copy
import re
from collections import Counter as CollectionCounter, defaultdict, deque
from collections.abc import Callable, Hashable as CollectionsHashable, Iterable as CollectionsIterable
from typing import (
from typing_extensions import Annotated, Final
from . import errors as errors_
from .class_validators import Validator, make_generic_validator, prep_validators
from .error_wrappers import ErrorWrapper
from .errors import ConfigError, InvalidDiscriminator, MissingDiscriminator, NoneIsNotAllowedError
from .types import Json, JsonWrapper
from .typing import (
from .utils import (
from .validators import constant_validator, dict_validator, find_validators, validate_json
def _type_analysis(self) -> None:
    if lenient_issubclass(self.type_, JsonWrapper):
        self.type_ = self.type_.inner_type
        self.parse_json = True
    elif lenient_issubclass(self.type_, Json):
        self.type_ = Any
        self.parse_json = True
    elif isinstance(self.type_, TypeVar):
        if self.type_.__bound__:
            self.type_ = self.type_.__bound__
        elif self.type_.__constraints__:
            self.type_ = Union[self.type_.__constraints__]
        else:
            self.type_ = Any
    elif is_new_type(self.type_):
        self.type_ = new_type_supertype(self.type_)
    if self.type_ is Any or self.type_ is object:
        if self.required is Undefined:
            self.required = False
        self.allow_none = True
        return
    elif self.type_ is Pattern or self.type_ is re.Pattern:
        return
    elif is_literal_type(self.type_):
        return
    elif is_typeddict(self.type_):
        return
    if is_finalvar(self.type_):
        self.final = True
        if self.type_ is Final:
            self.type_ = Any
        else:
            self.type_ = get_args(self.type_)[0]
        self._type_analysis()
        return
    origin = get_origin(self.type_)
    if origin is Annotated or is_typeddict_special(origin):
        self.type_ = get_args(self.type_)[0]
        self._type_analysis()
        return
    if self.discriminator_key is not None and (not is_union(origin)):
        raise TypeError('`discriminator` can only be used with `Union` type with more than one variant')
    if origin is None or origin is CollectionsHashable:
        if isinstance(self.type_, type) and isinstance(None, self.type_):
            self.allow_none = True
        return
    elif origin is Callable:
        return
    elif is_union(origin):
        types_ = []
        for type_ in get_args(self.type_):
            if is_none_type(type_) or type_ is Any or type_ is object:
                if self.required is Undefined:
                    self.required = False
                self.allow_none = True
            if is_none_type(type_):
                continue
            types_.append(type_)
        if len(types_) == 1:
            self.type_ = types_[0]
            self.outer_type_ = self.type_
            self._type_analysis()
        else:
            self.sub_fields = [self._create_sub_type(t, f'{self.name}_{display_as_type(t)}') for t in types_]
            if self.discriminator_key is not None:
                self.prepare_discriminated_union_sub_fields()
        return
    elif issubclass(origin, Tuple):
        args = get_args(self.type_)
        if not args:
            self.type_ = Any
            self.shape = SHAPE_TUPLE_ELLIPSIS
        elif len(args) == 2 and args[1] is Ellipsis:
            self.type_ = args[0]
            self.shape = SHAPE_TUPLE_ELLIPSIS
            self.sub_fields = [self._create_sub_type(args[0], f'{self.name}_0')]
        elif args == ((),):
            self.shape = SHAPE_TUPLE
            self.type_ = Any
            self.sub_fields = []
        else:
            self.shape = SHAPE_TUPLE
            self.sub_fields = [self._create_sub_type(t, f'{self.name}_{i}') for i, t in enumerate(args)]
        return
    elif issubclass(origin, List):
        get_validators = getattr(self.type_, '__get_validators__', None)
        if get_validators:
            self.class_validators.update({f'list_{i}': Validator(validator, pre=True) for i, validator in enumerate(get_validators())})
        self.type_ = get_args(self.type_)[0]
        self.shape = SHAPE_LIST
    elif issubclass(origin, Set):
        get_validators = getattr(self.type_, '__get_validators__', None)
        if get_validators:
            self.class_validators.update({f'set_{i}': Validator(validator, pre=True) for i, validator in enumerate(get_validators())})
        self.type_ = get_args(self.type_)[0]
        self.shape = SHAPE_SET
    elif issubclass(origin, FrozenSet):
        get_validators = getattr(self.type_, '__get_validators__', None)
        if get_validators:
            self.class_validators.update({f'frozenset_{i}': Validator(validator, pre=True) for i, validator in enumerate(get_validators())})
        self.type_ = get_args(self.type_)[0]
        self.shape = SHAPE_FROZENSET
    elif issubclass(origin, Deque):
        self.type_ = get_args(self.type_)[0]
        self.shape = SHAPE_DEQUE
    elif issubclass(origin, Sequence):
        self.type_ = get_args(self.type_)[0]
        self.shape = SHAPE_SEQUENCE
    elif origin is dict or origin is Dict:
        self.key_field = self._create_sub_type(get_args(self.type_)[0], 'key_' + self.name, for_keys=True)
        self.type_ = get_args(self.type_)[1]
        self.shape = SHAPE_DICT
    elif issubclass(origin, DefaultDict):
        self.key_field = self._create_sub_type(get_args(self.type_)[0], 'key_' + self.name, for_keys=True)
        self.type_ = get_args(self.type_)[1]
        self.shape = SHAPE_DEFAULTDICT
    elif issubclass(origin, Counter):
        self.key_field = self._create_sub_type(get_args(self.type_)[0], 'key_' + self.name, for_keys=True)
        self.type_ = int
        self.shape = SHAPE_COUNTER
    elif issubclass(origin, Mapping):
        self.key_field = self._create_sub_type(get_args(self.type_)[0], 'key_' + self.name, for_keys=True)
        self.type_ = get_args(self.type_)[1]
        self.shape = SHAPE_MAPPING
    elif origin in {Iterable, CollectionsIterable}:
        self.type_ = get_args(self.type_)[0]
        self.shape = SHAPE_ITERABLE
        self.sub_fields = [self._create_sub_type(self.type_, f'{self.name}_type')]
    elif issubclass(origin, Type):
        return
    elif hasattr(origin, '__get_validators__') or self.model_config.arbitrary_types_allowed:
        self.shape = SHAPE_GENERIC
        self.sub_fields = [self._create_sub_type(t, f'{self.name}_{i}') for i, t in enumerate(get_args(self.type_))]
        self.type_ = origin
        return
    else:
        raise TypeError(f'Fields of type "{origin}" are not supported.')
    self.sub_fields = [self._create_sub_type(self.type_, '_' + self.name)]