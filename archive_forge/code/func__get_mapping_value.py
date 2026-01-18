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
def _get_mapping_value(self, original: T, converted: Dict[Any, Any]) -> Union[T, Dict[Any, Any]]:
    """
        When type is `Mapping[KT, KV]` (or another unsupported mapping), we try to avoid
        coercing to `dict` unwillingly.
        """
    original_cls = original.__class__
    if original_cls == dict or original_cls == Dict:
        return converted
    elif original_cls in {defaultdict, DefaultDict}:
        return defaultdict(self.type_, converted)
    else:
        try:
            return original_cls(converted)
        except TypeError:
            raise RuntimeError(f'Could not convert dictionary to {original_cls.__name__!r}') from None