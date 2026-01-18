from typing import (
from tenacity import (
from langchain_core.runnables.base import Input, Output, RunnableBindingBase
from langchain_core.runnables.config import RunnableConfig, patch_config
@property
def _kwargs_retrying(self) -> Dict[str, Any]:
    kwargs: Dict[str, Any] = dict()
    if self.max_attempt_number:
        kwargs['stop'] = stop_after_attempt(self.max_attempt_number)
    if self.wait_exponential_jitter:
        kwargs['wait'] = wait_exponential_jitter()
    if self.retry_exception_types:
        kwargs['retry'] = retry_if_exception_type(self.retry_exception_types)
    return kwargs