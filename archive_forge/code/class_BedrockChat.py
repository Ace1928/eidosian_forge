import re
from collections import defaultdict
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union
from langchain_core._api.deprecation import deprecated
from langchain_core.callbacks import (
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.pydantic_v1 import Extra
from langchain_community.chat_models.anthropic import (
from langchain_community.chat_models.meta import convert_messages_to_prompt_llama
from langchain_community.llms.bedrock import BedrockBase
from langchain_community.utilities.anthropic import (
@deprecated(since='0.0.34', removal='0.3', alternative_import='langchain_aws.ChatBedrock')
class BedrockChat(BaseChatModel, BedrockBase):
    """Chat model that uses the Bedrock API."""

    @property
    def _llm_type(self) -> str:
        """Return type of chat model."""
        return 'amazon_bedrock_chat'

    @classmethod
    def is_lc_serializable(cls) -> bool:
        """Return whether this model can be serialized by Langchain."""
        return True

    @classmethod
    def get_lc_namespace(cls) -> List[str]:
        """Get the namespace of the langchain object."""
        return ['langchain', 'chat_models', 'bedrock']

    @property
    def lc_attributes(self) -> Dict[str, Any]:
        attributes: Dict[str, Any] = {}
        if self.region_name:
            attributes['region_name'] = self.region_name
        return attributes

    class Config:
        """Configuration for this pydantic object."""
        extra = Extra.forbid

    def _stream(self, messages: List[BaseMessage], stop: Optional[List[str]]=None, run_manager: Optional[CallbackManagerForLLMRun]=None, **kwargs: Any) -> Iterator[ChatGenerationChunk]:
        provider = self._get_provider()
        prompt, system, formatted_messages = (None, None, None)
        if provider == 'anthropic':
            system, formatted_messages = ChatPromptAdapter.format_messages(provider, messages)
        else:
            prompt = ChatPromptAdapter.convert_messages_to_prompt(provider=provider, messages=messages)
        for chunk in self._prepare_input_and_invoke_stream(prompt=prompt, system=system, messages=formatted_messages, stop=stop, run_manager=run_manager, **kwargs):
            delta = chunk.text
            yield ChatGenerationChunk(message=AIMessageChunk(content=delta))

    def _generate(self, messages: List[BaseMessage], stop: Optional[List[str]]=None, run_manager: Optional[CallbackManagerForLLMRun]=None, **kwargs: Any) -> ChatResult:
        completion = ''
        llm_output: Dict[str, Any] = {'model_id': self.model_id}
        if self.streaming:
            for chunk in self._stream(messages, stop, run_manager, **kwargs):
                completion += chunk.text
        else:
            provider = self._get_provider()
            prompt, system, formatted_messages = (None, None, None)
            params: Dict[str, Any] = {**kwargs}
            if provider == 'anthropic':
                system, formatted_messages = ChatPromptAdapter.format_messages(provider, messages)
            else:
                prompt = ChatPromptAdapter.convert_messages_to_prompt(provider=provider, messages=messages)
            if stop:
                params['stop_sequences'] = stop
            completion, usage_info = self._prepare_input_and_invoke(prompt=prompt, stop=stop, run_manager=run_manager, system=system, messages=formatted_messages, **params)
            llm_output['usage'] = usage_info
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=completion))], llm_output=llm_output)

    def _combine_llm_outputs(self, llm_outputs: List[Optional[dict]]) -> dict:
        final_usage: Dict[str, int] = defaultdict(int)
        final_output = {}
        for output in llm_outputs:
            output = output or {}
            usage = output.get('usage', {})
            for token_type, token_count in usage.items():
                final_usage[token_type] += token_count
            final_output.update(output)
        final_output['usage'] = final_usage
        return final_output

    def get_num_tokens(self, text: str) -> int:
        if self._model_is_anthropic:
            return get_num_tokens_anthropic(text)
        else:
            return super().get_num_tokens(text)

    def get_token_ids(self, text: str) -> List[int]:
        if self._model_is_anthropic:
            return get_token_ids_anthropic(text)
        else:
            return super().get_token_ids(text)