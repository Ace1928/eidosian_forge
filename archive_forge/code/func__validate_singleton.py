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
def _validate_singleton(self, v: Any, values: Dict[str, Any], loc: 'LocStr', cls: Optional['ModelOrDc']) -> 'ValidateReturn':
    if self.sub_fields:
        if self.discriminator_key is not None:
            return self._validate_discriminated_union(v, values, loc, cls)
        errors = []
        if self.model_config.smart_union and is_union(get_origin(self.type_)):
            for field in self.sub_fields:
                if v.__class__ is field.outer_type_:
                    return (v, None)
            for field in self.sub_fields:
                try:
                    if isinstance(v, field.outer_type_):
                        return (v, None)
                except TypeError:
                    if lenient_isinstance(v, get_origin(field.outer_type_)):
                        value, error = field.validate(v, values, loc=loc, cls=cls)
                        if not error:
                            return (value, None)
        for field in self.sub_fields:
            value, error = field.validate(v, values, loc=loc, cls=cls)
            if error:
                errors.append(error)
            else:
                return (value, None)
        return (v, errors)
    else:
        return self._apply_validators(v, values, loc, cls, self.validators)