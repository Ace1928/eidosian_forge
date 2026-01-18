import re
import warnings
from typing import (
from langchain_core._api.deprecation import deprecated
from langchain_core.callbacks import (
from langchain_core.language_models import BaseLanguageModel
from langchain_core.language_models.llms import LLM
from langchain_core.outputs import GenerationChunk
from langchain_core.prompt_values import PromptValue
from langchain_core.pydantic_v1 import Field, SecretStr, root_validator
from langchain_core.utils import (
from langchain_core.utils.utils import build_extra_kwargs, convert_to_secret_str
class _AnthropicCommon(BaseLanguageModel):
    client: Any = None
    async_client: Any = None
    model: str = Field(default='claude-2', alias='model_name')
    'Model name to use.'
    max_tokens_to_sample: int = Field(default=256, alias='max_tokens')
    'Denotes the number of tokens to predict per generation.'
    temperature: Optional[float] = None
    'A non-negative float that tunes the degree of randomness in generation.'
    top_k: Optional[int] = None
    'Number of most likely tokens to consider at each step.'
    top_p: Optional[float] = None
    'Total probability mass of tokens to consider at each step.'
    streaming: bool = False
    'Whether to stream the results.'
    default_request_timeout: Optional[float] = None
    'Timeout for requests to Anthropic Completion API. Default is 600 seconds.'
    max_retries: int = 2
    'Number of retries allowed for requests sent to the Anthropic Completion API.'
    anthropic_api_url: Optional[str] = None
    anthropic_api_key: Optional[SecretStr] = None
    HUMAN_PROMPT: Optional[str] = None
    AI_PROMPT: Optional[str] = None
    count_tokens: Optional[Callable[[str], int]] = None
    model_kwargs: Dict[str, Any] = Field(default_factory=dict)

    @root_validator(pre=True)
    def build_extra(cls, values: Dict) -> Dict:
        extra = values.get('model_kwargs', {})
        all_required_field_names = get_pydantic_field_names(cls)
        values['model_kwargs'] = build_extra_kwargs(extra, values, all_required_field_names)
        return values

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that api key and python package exists in environment."""
        values['anthropic_api_key'] = convert_to_secret_str(get_from_dict_or_env(values, 'anthropic_api_key', 'ANTHROPIC_API_KEY'))
        values['anthropic_api_url'] = get_from_dict_or_env(values, 'anthropic_api_url', 'ANTHROPIC_API_URL', default='https://api.anthropic.com')
        try:
            import anthropic
            check_package_version('anthropic', gte_version='0.3')
            values['client'] = anthropic.Anthropic(base_url=values['anthropic_api_url'], api_key=values['anthropic_api_key'].get_secret_value(), timeout=values['default_request_timeout'], max_retries=values['max_retries'])
            values['async_client'] = anthropic.AsyncAnthropic(base_url=values['anthropic_api_url'], api_key=values['anthropic_api_key'].get_secret_value(), timeout=values['default_request_timeout'], max_retries=values['max_retries'])
            values['HUMAN_PROMPT'] = anthropic.HUMAN_PROMPT
            values['AI_PROMPT'] = anthropic.AI_PROMPT
            values['count_tokens'] = values['client'].count_tokens
        except ImportError:
            raise ImportError('Could not import anthropic python package. Please it install it with `pip install anthropic`.')
        return values

    @property
    def _default_params(self) -> Mapping[str, Any]:
        """Get the default parameters for calling Anthropic API."""
        d = {'max_tokens_to_sample': self.max_tokens_to_sample, 'model': self.model}
        if self.temperature is not None:
            d['temperature'] = self.temperature
        if self.top_k is not None:
            d['top_k'] = self.top_k
        if self.top_p is not None:
            d['top_p'] = self.top_p
        return {**d, **self.model_kwargs}

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {**{}, **self._default_params}

    def _get_anthropic_stop(self, stop: Optional[List[str]]=None) -> List[str]:
        if not self.HUMAN_PROMPT or not self.AI_PROMPT:
            raise NameError('Please ensure the anthropic package is loaded')
        if stop is None:
            stop = []
        stop.extend([self.HUMAN_PROMPT])
        return stop