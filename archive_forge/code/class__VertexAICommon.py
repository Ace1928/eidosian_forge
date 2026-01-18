from __future__ import annotations
from concurrent.futures import Executor, ThreadPoolExecutor
from typing import TYPE_CHECKING, Any, ClassVar, Dict, Iterator, List, Optional, Union
from langchain_core._api.deprecation import deprecated
from langchain_core.callbacks.manager import (
from langchain_core.language_models.llms import BaseLLM
from langchain_core.outputs import Generation, GenerationChunk, LLMResult
from langchain_core.pydantic_v1 import BaseModel, Field, root_validator
from langchain_community.utilities.vertexai import (
class _VertexAICommon(_VertexAIBase):
    client: '_LanguageModel' = None
    client_preview: '_LanguageModel' = None
    model_name: str
    'Underlying model name.'
    temperature: float = 0.0
    'Sampling temperature, it controls the degree of randomness in token selection.'
    max_output_tokens: int = 128
    'Token limit determines the maximum amount of text output from one prompt.'
    top_p: float = 0.95
    'Tokens are selected from most probable to least until the sum of their '
    'probabilities equals the top-p value. Top-p is ignored for Codey models.'
    top_k: int = 40
    'How the model selects tokens for output, the next token is selected from '
    'among the top-k most probable tokens. Top-k is ignored for Codey models.'
    credentials: Any = Field(default=None, exclude=True)
    'The default custom credentials (google.auth.credentials.Credentials) to use '
    'when making API calls. If not provided, credentials will be ascertained from '
    'the environment.'
    n: int = 1
    'How many completions to generate for each prompt.'
    streaming: bool = False
    'Whether to stream the results or not.'

    @property
    def _llm_type(self) -> str:
        return 'vertexai'

    @property
    def is_codey_model(self) -> bool:
        return is_codey_model(self.model_name)

    @property
    def _is_gemini_model(self) -> bool:
        return is_gemini_model(self.model_name)

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """Gets the identifying parameters."""
        return {**{'model_name': self.model_name}, **self._default_params}

    @property
    def _default_params(self) -> Dict[str, Any]:
        params = {'temperature': self.temperature, 'max_output_tokens': self.max_output_tokens, 'candidate_count': self.n}
        if not self.is_codey_model:
            params.update({'top_k': self.top_k, 'top_p': self.top_p})
        return params

    @classmethod
    def _try_init_vertexai(cls, values: Dict) -> None:
        allowed_params = ['project', 'location', 'credentials']
        params = {k: v for k, v in values.items() if k in allowed_params}
        init_vertexai(**params)
        return None

    def _prepare_params(self, stop: Optional[List[str]]=None, stream: bool=False, **kwargs: Any) -> dict:
        stop_sequences = stop or self.stop
        params_mapping = {'n': 'candidate_count'}
        params = {params_mapping.get(k, k): v for k, v in kwargs.items()}
        params = {**self._default_params, 'stop_sequences': stop_sequences, **params}
        if stream or self.streaming:
            params.pop('candidate_count')
        return params