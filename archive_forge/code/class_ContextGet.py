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
@beta()
class ContextGet(RunnableSerializable):
    """Get a context value."""
    prefix: str = ''
    key: Union[str, List[str]]

    def __str__(self) -> str:
        return f'ContextGet({_print_keys(self.key)})'

    @property
    def ids(self) -> List[str]:
        prefix = self.prefix + '/' if self.prefix else ''
        keys = self.key if isinstance(self.key, list) else [self.key]
        return [f'{CONTEXT_CONFIG_PREFIX}{prefix}{k}{CONTEXT_CONFIG_SUFFIX_GET}' for k in keys]

    @property
    def config_specs(self) -> List[ConfigurableFieldSpec]:
        return super().config_specs + [ConfigurableFieldSpec(id=id_, annotation=Callable[[], Any]) for id_ in self.ids]

    def invoke(self, input: Any, config: Optional[RunnableConfig]=None) -> Any:
        config = ensure_config(config)
        configurable = config.get('configurable', {})
        if isinstance(self.key, list):
            return {key: configurable[id_]() for key, id_ in zip(self.key, self.ids)}
        else:
            return configurable[self.ids[0]]()

    async def ainvoke(self, input: Any, config: Optional[RunnableConfig]=None, **kwargs: Any) -> Any:
        config = ensure_config(config)
        configurable = config.get('configurable', {})
        if isinstance(self.key, list):
            values = await asyncio.gather(*(configurable[id_]() for id_ in self.ids))
            return {key: value for key, value in zip(self.key, values)}
        else:
            return await configurable[self.ids[0]]()