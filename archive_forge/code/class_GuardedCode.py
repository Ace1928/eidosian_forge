import dataclasses
import sys
import types
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Protocol, Union
from typing_extensions import TypeAlias
import torch
@dataclasses.dataclass
class GuardedCode:
    code: types.CodeType
    check_fn: GuardFn