from typing import Any, Iterator, List, Optional
from langchain_core.callbacks.manager import (
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
from langchain_core.outputs import (
from langchain_community.llms.mlx_pipeline import MLXPipeline
def _to_chat_prompt(self, messages: List[BaseMessage], tokenize: bool=False, return_tensors: Optional[str]=None) -> str:
    """Convert a list of messages into a prompt format expected by wrapped LLM."""
    if not messages:
        raise ValueError('At least one HumanMessage must be provided!')
    if not isinstance(messages[-1], HumanMessage):
        raise ValueError('Last message must be a HumanMessage!')
    messages_dicts = [self._to_chatml_format(m) for m in messages]
    return self.tokenizer.apply_chat_template(messages_dicts, tokenize=tokenize, add_generation_prompt=True, return_tensors=return_tensors)