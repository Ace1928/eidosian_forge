import asyncio
import threading
from collections import defaultdict
from functools import partial
from itertools import groupby
from typing import (
from langchain_core._api.beta_decorator import beta
from langchain_core.runnables.base import (
from langchain_core.runnables.config import RunnableConfig, ensure_config, patch_config
from langchain_core.runnables.utils import ConfigurableFieldSpec, Input, Output
class PrefixContext:
    """Context for a runnable with a prefix."""
    prefix: str = ''

    def __init__(self, prefix: str=''):
        self.prefix = prefix

    def getter(self, key: Union[str, List[str]], /) -> ContextGet:
        return ContextGet(key=key, prefix=self.prefix)

    def setter(self, _key: Optional[str]=None, _value: Optional[SetValue]=None, /, **kwargs: SetValue) -> ContextSet:
        return ContextSet(_key, _value, prefix=self.prefix, **kwargs)