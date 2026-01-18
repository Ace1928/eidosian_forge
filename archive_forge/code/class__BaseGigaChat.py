from __future__ import annotations
import logging
from functools import cached_property
from typing import TYPE_CHECKING, Any, AsyncIterator, Dict, Iterator, List, Optional
from langchain_core.callbacks import (
from langchain_core.language_models.llms import BaseLLM
from langchain_core.load.serializable import Serializable
from langchain_core.outputs import Generation, GenerationChunk, LLMResult
from langchain_core.pydantic_v1 import root_validator
class _BaseGigaChat(Serializable):
    base_url: Optional[str] = None
    ' Base API URL '
    auth_url: Optional[str] = None
    ' Auth URL '
    credentials: Optional[str] = None
    ' Auth Token '
    scope: Optional[str] = None
    ' Permission scope for access token '
    access_token: Optional[str] = None
    ' Access token for GigaChat '
    model: Optional[str] = None
    'Model name to use.'
    user: Optional[str] = None
    ' Username for authenticate '
    password: Optional[str] = None
    ' Password for authenticate '
    timeout: Optional[float] = None
    ' Timeout for request '
    verify_ssl_certs: Optional[bool] = None
    ' Check certificates for all requests '
    ca_bundle_file: Optional[str] = None
    cert_file: Optional[str] = None
    key_file: Optional[str] = None
    key_file_password: Optional[str] = None
    profanity: bool = True
    ' DEPRECATED: Check for profanity '
    profanity_check: Optional[bool] = None
    ' Check for profanity '
    streaming: bool = False
    ' Whether to stream the results or not. '
    temperature: Optional[float] = None
    ' What sampling temperature to use. '
    max_tokens: Optional[int] = None
    ' Maximum number of tokens to generate '
    use_api_for_tokens: bool = False
    ' Use GigaChat API for tokens count '
    verbose: bool = False
    ' Verbose logging '
    top_p: Optional[float] = None
    ' top_p value to use for nucleus sampling. Must be between 0.0 and 1.0 '
    repetition_penalty: Optional[float] = None
    ' The penalty applied to repeated tokens '
    update_interval: Optional[float] = None
    ' Minimum interval in seconds that elapses between sending tokens '

    @property
    def _llm_type(self) -> str:
        return 'giga-chat-model'

    @property
    def lc_secrets(self) -> Dict[str, str]:
        return {'credentials': 'GIGACHAT_CREDENTIALS', 'access_token': 'GIGACHAT_ACCESS_TOKEN', 'password': 'GIGACHAT_PASSWORD', 'key_file_password': 'GIGACHAT_KEY_FILE_PASSWORD'}

    @property
    def lc_serializable(self) -> bool:
        return True

    @cached_property
    def _client(self) -> gigachat.GigaChat:
        """Returns GigaChat API client"""
        import gigachat
        return gigachat.GigaChat(base_url=self.base_url, auth_url=self.auth_url, credentials=self.credentials, scope=self.scope, access_token=self.access_token, model=self.model, profanity_check=self.profanity_check, user=self.user, password=self.password, timeout=self.timeout, verify_ssl_certs=self.verify_ssl_certs, ca_bundle_file=self.ca_bundle_file, cert_file=self.cert_file, key_file=self.key_file, key_file_password=self.key_file_password, verbose=self.verbose)

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate authenticate data in environment and python package is installed."""
        try:
            import gigachat
        except ImportError:
            raise ImportError('Could not import gigachat python package. Please install it with `pip install gigachat`.')
        fields = set(cls.__fields__.keys())
        diff = set(values.keys()) - fields
        if diff:
            logger.warning(f'Extra fields {diff} in GigaChat class')
        if 'profanity' in fields and values.get('profanity') is False:
            logger.warning("'profanity' field is deprecated. Use 'profanity_check' instead.")
            if values.get('profanity_check') is None:
                values['profanity_check'] = values.get('profanity')
        return values

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """Get the identifying parameters."""
        return {'temperature': self.temperature, 'model': self.model, 'profanity': self.profanity_check, 'streaming': self.streaming, 'max_tokens': self.max_tokens, 'top_p': self.top_p, 'repetition_penalty': self.repetition_penalty}

    def tokens_count(self, input_: List[str], model: Optional[str]=None) -> List[gm.TokensCount]:
        """Get tokens of string list"""
        return self._client.tokens_count(input_, model)

    async def atokens_count(self, input_: List[str], model: Optional[str]=None) -> List[gm.TokensCount]:
        """Get tokens of strings list (async)"""
        return await self._client.atokens_count(input_, model)

    def get_models(self) -> gm.Models:
        """Get available models of Gigachat"""
        return self._client.get_models()

    async def aget_models(self) -> gm.Models:
        """Get available models of Gigachat (async)"""
        return await self._client.aget_models()

    def get_model(self, model: str) -> gm.Model:
        """Get info about model"""
        return self._client.get_model(model)

    async def aget_model(self, model: str) -> gm.Model:
        """Get info about model (async)"""
        return await self._client.aget_model(model)

    def get_num_tokens(self, text: str) -> int:
        """Count approximate number of tokens"""
        if self.use_api_for_tokens:
            return self.tokens_count([text])[0].tokens
        else:
            return round(len(text) / 4.6)