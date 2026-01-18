import threading
from typing import Any, Dict, List
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult
class OpenAICallbackHandler(BaseCallbackHandler):
    """Callback Handler that tracks OpenAI info."""
    total_tokens: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    successful_requests: int = 0
    total_cost: float = 0.0

    def __init__(self) -> None:
        super().__init__()
        self._lock = threading.Lock()

    def __repr__(self) -> str:
        return f'Tokens Used: {self.total_tokens}\n\tPrompt Tokens: {self.prompt_tokens}\n\tCompletion Tokens: {self.completion_tokens}\nSuccessful Requests: {self.successful_requests}\nTotal Cost (USD): ${self.total_cost}'

    @property
    def always_verbose(self) -> bool:
        """Whether to call verbose callbacks even if verbose is False."""
        return True

    def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any) -> None:
        """Print out the prompts."""
        pass

    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        """Print out the token."""
        pass

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        """Collect token usage."""
        if response.llm_output is None:
            return None
        if 'token_usage' not in response.llm_output:
            with self._lock:
                self.successful_requests += 1
            return None
        token_usage = response.llm_output['token_usage']
        completion_tokens = token_usage.get('completion_tokens', 0)
        prompt_tokens = token_usage.get('prompt_tokens', 0)
        model_name = standardize_model_name(response.llm_output.get('model_name', ''))
        if model_name in MODEL_COST_PER_1K_TOKENS:
            completion_cost = get_openai_token_cost_for_model(model_name, completion_tokens, is_completion=True)
            prompt_cost = get_openai_token_cost_for_model(model_name, prompt_tokens)
        else:
            completion_cost = 0
            prompt_cost = 0
        with self._lock:
            self.total_cost += prompt_cost + completion_cost
            self.total_tokens += token_usage.get('total_tokens', 0)
            self.prompt_tokens += prompt_tokens
            self.completion_tokens += completion_tokens
            self.successful_requests += 1

    def __copy__(self) -> 'OpenAICallbackHandler':
        """Return a copy of the callback handler."""
        return self

    def __deepcopy__(self, memo: Any) -> 'OpenAICallbackHandler':
        """Return a deep copy of the callback handler."""
        return self