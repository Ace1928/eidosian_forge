from __future__ import annotations
import logging
from typing import (
from langchain_core.callbacks import (
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.language_models.llms import create_base_retry_decorator
from langchain_core.messages import (
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.pydantic_v1 import BaseModel, Extra, SecretStr, root_validator
from langchain_core.utils import get_from_dict_or_env
class ChatPremAI(BaseChatModel, BaseModel):
    """PremAI Chat models.

    To use, you will need to have an API key. You can find your existing API Key
    or generate a new one here: https://app.premai.io/api_keys/
    """
    project_id: int
    'The project ID in which the experiments or deployments are carried out. \n    You can find all your projects here: https://app.premai.io/projects/'
    premai_api_key: Optional[SecretStr] = None
    'Prem AI API Key. Get it here: https://app.premai.io/api_keys/'
    model: Optional[str] = None
    "Name of the model. This is an optional parameter. \n    The default model is the one deployed from Prem's LaunchPad: https://app.premai.io/projects/8/launchpad\n    If model name is other than default model then it will override the calls \n    from the model deployed from launchpad."
    session_id: Optional[str] = None
    'The ID of the session to use. It helps to track the chat history.'
    temperature: Optional[float] = None
    'Model temperature. Value should be >= 0 and <= 1.0'
    top_p: Optional[float] = None
    'top_p adjusts the number of choices for each predicted tokens based on\n        cumulative probabilities. Value should be ranging between 0.0 and 1.0. \n    '
    max_tokens: Optional[int] = None
    'The maximum number of tokens to generate'
    max_retries: int = 1
    'Max number of retries to call the API'
    system_prompt: Optional[str] = ''
    "Acts like a default instruction that helps the LLM act or generate \n    in a specific way.This is an Optional Parameter. By default the \n    system prompt would be using Prem's Launchpad models system prompt. \n    Changing the system prompt would override the default system prompt.\n    "
    streaming: Optional[bool] = False
    'Whether to stream the responses or not.'
    tools: Optional[Dict[str, Any]] = None
    'A list of tools the model may call. Currently, only functions are \n    supported as a tool'
    frequency_penalty: Optional[float] = None
    'Number between -2.0 and 2.0. Positive values penalize new tokens based'
    presence_penalty: Optional[float] = None
    'Number between -2.0 and 2.0. Positive values penalize new tokens based \n    on whether they appear in the text so far.'
    logit_bias: Optional[dict] = None
    'JSON object that maps tokens to an associated bias value from -100 to 100.'
    stop: Optional[Union[str, List[str]]] = None
    'Up to 4 sequences where the API will stop generating further tokens.'
    seed: Optional[int] = None
    'This feature is in Beta. If specified, our system will make a best effort \n    to sample deterministically.'
    client: Any

    class Config:
        """Configuration for this pydantic object."""
        extra = Extra.forbid

    @root_validator()
    def validate_environments(cls, values: Dict) -> Dict:
        """Validate that the package is installed and that the API token is valid"""
        try:
            from premai import Prem
        except ImportError as error:
            raise ImportError('Could not import Prem Python package.Please install it with: `pip install premai`') from error
        try:
            premai_api_key = get_from_dict_or_env(values, 'premai_api_key', 'PREMAI_API_KEY')
            values['client'] = Prem(api_key=premai_api_key)
        except Exception as error:
            raise ValueError('Your API Key is incorrect. Please try again.') from error
        return values

    @property
    def _llm_type(self) -> str:
        return 'premai'

    @property
    def _default_params(self) -> Dict[str, Any]:
        return {'model': self.model, 'system_prompt': self.system_prompt, 'top_p': self.top_p, 'temperature': self.temperature, 'logit_bias': self.logit_bias, 'max_tokens': self.max_tokens, 'presence_penalty': self.presence_penalty, 'frequency_penalty': self.frequency_penalty, 'seed': self.seed, 'stop': None}

    def _get_all_kwargs(self, **kwargs: Any) -> Dict[str, Any]:
        all_kwargs = {**self._default_params, **kwargs}
        for key in list(self._default_params.keys()):
            if all_kwargs.get(key) is None or all_kwargs.get(key) == '':
                all_kwargs.pop(key, None)
        return all_kwargs

    def _generate(self, messages: List[BaseMessage], stop: Optional[List[str]]=None, run_manager: Optional[CallbackManagerForLLMRun]=None, **kwargs: Any) -> ChatResult:
        system_prompt, messages_to_pass = _messages_to_prompt_dict(messages)
        kwargs['stop'] = stop
        if system_prompt is not None and system_prompt != '':
            kwargs['system_prompt'] = system_prompt
        all_kwargs = self._get_all_kwargs(**kwargs)
        response = chat_with_retry(self, project_id=self.project_id, messages=messages_to_pass, stream=False, run_manager=run_manager, **all_kwargs)
        return _response_to_result(response=response, stop=stop)

    def _stream(self, messages: List[BaseMessage], stop: Optional[List[str]]=None, run_manager: Optional[CallbackManagerForLLMRun]=None, **kwargs: Any) -> Iterator[ChatGenerationChunk]:
        system_prompt, messages_to_pass = _messages_to_prompt_dict(messages)
        kwargs['stop'] = stop
        if 'system_prompt' not in kwargs:
            if system_prompt is not None and system_prompt != '':
                kwargs['system_prompt'] = system_prompt
        all_kwargs = self._get_all_kwargs(**kwargs)
        default_chunk_class = AIMessageChunk
        for streamed_response in chat_with_retry(self, project_id=self.project_id, messages=messages_to_pass, stream=True, run_manager=run_manager, **all_kwargs):
            try:
                chunk, finish_reason = _convert_delta_response_to_message_chunk(response=streamed_response, default_class=default_chunk_class)
                generation_info = dict(finish_reason=finish_reason) if finish_reason is not None else None
                cg_chunk = ChatGenerationChunk(message=chunk, generation_info=generation_info)
                if run_manager:
                    run_manager.on_llm_new_token(cg_chunk.text, chunk=cg_chunk)
                yield cg_chunk
            except Exception as _:
                continue