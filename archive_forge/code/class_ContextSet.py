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
class ContextSet(RunnableSerializable):
    """Set a context value."""
    prefix: str = ''
    keys: Mapping[str, Optional[Runnable]]

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, key: Optional[str]=None, value: Optional[SetValue]=None, prefix: str='', **kwargs: SetValue):
        if key is not None:
            kwargs[key] = value
        super().__init__(keys={k: _coerce_set_value(v) if v is not None else None for k, v in kwargs.items()}, prefix=prefix)

    def __str__(self) -> str:
        return f'ContextSet({_print_keys(list(self.keys.keys()))})'

    @property
    def ids(self) -> List[str]:
        prefix = self.prefix + '/' if self.prefix else ''
        return [f'{CONTEXT_CONFIG_PREFIX}{prefix}{key}{CONTEXT_CONFIG_SUFFIX_SET}' for key in self.keys]

    @property
    def config_specs(self) -> List[ConfigurableFieldSpec]:
        mapper_config_specs = [s for mapper in self.keys.values() if mapper is not None for s in mapper.config_specs]
        for spec in mapper_config_specs:
            if spec.id.endswith(CONTEXT_CONFIG_SUFFIX_GET):
                getter_key = spec.id.split('/')[1]
                if getter_key in self.keys:
                    raise ValueError(f'Circular reference in context setter for key {getter_key}')
        return super().config_specs + [ConfigurableFieldSpec(id=id_, annotation=Callable[[], Any]) for id_ in self.ids]

    def invoke(self, input: Any, config: Optional[RunnableConfig]=None) -> Any:
        config = ensure_config(config)
        configurable = config.get('configurable', {})
        for id_, mapper in zip(self.ids, self.keys.values()):
            if mapper is not None:
                configurable[id_](mapper.invoke(input, config))
            else:
                configurable[id_](input)
        return input

    async def ainvoke(self, input: Any, config: Optional[RunnableConfig]=None, **kwargs: Any) -> Any:
        config = ensure_config(config)
        configurable = config.get('configurable', {})
        for id_, mapper in zip(self.ids, self.keys.values()):
            if mapper is not None:
                await configurable[id_](await mapper.ainvoke(input, config))
            else:
                await configurable[id_](input)
        return input