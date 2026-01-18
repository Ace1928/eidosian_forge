import json
import warnings
from typing import (
from langchain_core.callbacks import (
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_community.llms.azureml_endpoint import (
class LlamaChatContentFormatter(CustomOpenAIChatContentFormatter):
    """Deprecated: Kept for backwards compatibility

    Chat Content formatter for Llama."""

    def __init__(self) -> None:
        super().__init__()
        warnings.warn('`LlamaChatContentFormatter` will be deprecated in the future. \n                Please use `CustomOpenAIChatContentFormatter` instead.  \n            ')