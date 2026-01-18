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
def _validate_discriminated_union(self, v: Any, values: Dict[str, Any], loc: 'LocStr', cls: Optional['ModelOrDc']) -> 'ValidateReturn':
    assert self.discriminator_key is not None
    assert self.discriminator_alias is not None
    try:
        try:
            discriminator_value = v[self.discriminator_alias]
        except KeyError:
            if self.model_config.allow_population_by_field_name:
                discriminator_value = v[self.discriminator_key]
            else:
                raise
    except KeyError:
        return (v, ErrorWrapper(MissingDiscriminator(discriminator_key=self.discriminator_key), loc))
    except TypeError:
        try:
            discriminator_value = getattr(v, self.discriminator_key)
        except (AttributeError, TypeError):
            return (v, ErrorWrapper(MissingDiscriminator(discriminator_key=self.discriminator_key), loc))
    if self.sub_fields_mapping is None:
        assert cls is not None
        raise ConfigError(f'field "{self.name}" not yet prepared so type is still a ForwardRef, you might need to call {cls.__name__}.update_forward_refs().')
    try:
        sub_field = self.sub_fields_mapping[discriminator_value]
    except (KeyError, TypeError):
        assert self.sub_fields_mapping is not None
        return (v, ErrorWrapper(InvalidDiscriminator(discriminator_key=self.discriminator_key, discriminator_value=discriminator_value, allowed_values=list(self.sub_fields_mapping)), loc))
    else:
        if not isinstance(loc, tuple):
            loc = (loc,)
        return sub_field.validate(v, values, loc=(*loc, display_as_type(sub_field.type_)), cls=cls)