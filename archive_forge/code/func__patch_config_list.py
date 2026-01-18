from typing import (
from tenacity import (
from langchain_core.runnables.base import Input, Output, RunnableBindingBase
from langchain_core.runnables.config import RunnableConfig, patch_config
def _patch_config_list(self, config: List[RunnableConfig], run_manager: List['T'], retry_state: RetryCallState) -> List[RunnableConfig]:
    return [self._patch_config(c, rm, retry_state) for c, rm in zip(config, run_manager)]