import keyword
import warnings
import weakref
from collections import OrderedDict, defaultdict, deque
from copy import deepcopy
from itertools import islice, zip_longest
from types import BuiltinFunctionType, CodeType, FunctionType, GeneratorType, LambdaType, ModuleType
from typing import (
from typing_extensions import Annotated
from .errors import ConfigError
from .typing import (
from .version import version_info
def _get_union_alias_and_all_values(union_type: Type[Any], discriminator_key: str) -> Tuple[str, Tuple[Tuple[str, ...], ...]]:
    zipped_aliases_values = [get_discriminator_alias_and_values(t, discriminator_key) for t in get_args(union_type)]
    all_aliases, all_values = zip(*zipped_aliases_values)
    return (get_unique_discriminator_alias(all_aliases, discriminator_key), all_values)