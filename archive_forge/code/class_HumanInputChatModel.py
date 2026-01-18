from io import StringIO
from typing import Any, Callable, Dict, List, Mapping, Optional
import yaml
from langchain_core.callbacks import (
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.pydantic_v1 import Field
from langchain_community.llms.utils import enforce_stop_tokens
class HumanInputChatModel(BaseChatModel):
    """ChatModel which returns user input as the response."""
    input_func: Callable = Field(default_factory=lambda: _collect_yaml_input)
    message_func: Callable = Field(default_factory=lambda: _display_messages)
    separator: str = '\n'
    input_kwargs: Mapping[str, Any] = {}
    message_kwargs: Mapping[str, Any] = {}

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        return {'input_func': self.input_func.__name__, 'message_func': self.message_func.__name__}

    @property
    def _llm_type(self) -> str:
        """Returns the type of LLM."""
        return 'human-input-chat-model'

    def _generate(self, messages: List[BaseMessage], stop: Optional[List[str]]=None, run_manager: Optional[CallbackManagerForLLMRun]=None, **kwargs: Any) -> ChatResult:
        """
        Displays the messages to the user and returns their input as a response.

        Args:
            messages (List[BaseMessage]): The messages to be displayed to the user.
            stop (Optional[List[str]]): A list of stop strings.
            run_manager (Optional[CallbackManagerForLLMRun]): Currently not used.

        Returns:
            ChatResult: The user's input as a response.
        """
        self.message_func(messages, **self.message_kwargs)
        user_input = self.input_func(messages, stop=stop, **self.input_kwargs)
        return ChatResult(generations=[ChatGeneration(message=user_input)])