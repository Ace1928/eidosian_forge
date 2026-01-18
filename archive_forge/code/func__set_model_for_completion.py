from typing import (
from langchain_core.callbacks.manager import (
from langchain_core.language_models.chat_models import (
from langchain_core.messages import (
from langchain_core.outputs import (
from langchain_community.chat_models.litellm import (
def _set_model_for_completion(self) -> None:
    self.model = self.router.model_list[0]['model_name']