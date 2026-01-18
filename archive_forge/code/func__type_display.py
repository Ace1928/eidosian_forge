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
def _type_display(self) -> PyObjectStr:
    t = display_as_type(self.type_)
    if self.shape in MAPPING_LIKE_SHAPES:
        t = f'Mapping[{display_as_type(self.key_field.type_)}, {t}]'
    elif self.shape == SHAPE_TUPLE:
        t = 'Tuple[{}]'.format(', '.join((display_as_type(f.type_) for f in self.sub_fields)))
    elif self.shape == SHAPE_GENERIC:
        assert self.sub_fields
        t = '{}[{}]'.format(display_as_type(self.type_), ', '.join((display_as_type(f.type_) for f in self.sub_fields)))
    elif self.shape != SHAPE_SINGLETON:
        t = SHAPE_NAME_LOOKUP[self.shape].format(t)
    if self.allow_none and (self.shape != SHAPE_SINGLETON or not self.sub_fields):
        t = f'Optional[{t}]'
    return PyObjectStr(t)