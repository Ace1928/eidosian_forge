from types import ModuleType, SimpleNamespace
from typing import TYPE_CHECKING, Any, Callable, Dict
from langchain_core.tracers import BaseTracer
def _initialize_comet_modules(self) -> None:
    comet_llm_api = import_comet_llm_api()
    self._chain: ModuleType = comet_llm_api.chain
    self._span: ModuleType = comet_llm_api.span
    self._chain_api: ModuleType = comet_llm_api.chain_api
    self._experiment_info: ModuleType = comet_llm_api.experiment_info
    self._flush: Callable[[], None] = comet_llm_api.flush