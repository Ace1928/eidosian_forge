from typing import (
from langchain_core.callbacks.manager import (
from langchain_core.language_models.chat_models import (
from langchain_core.messages import (
from langchain_core.outputs import (
from langchain_community.chat_models.litellm import (
def _prepare_params_for_router(self, params: Any) -> None:
    params['model'] = self.model
    api_base_key_name = 'api_base'
    if api_base_key_name in params and params[api_base_key_name] is None:
        del params[api_base_key_name]
    params.setdefault('metadata', {})