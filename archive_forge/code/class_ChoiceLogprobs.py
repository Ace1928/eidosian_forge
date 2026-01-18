from typing import List, Optional
from typing_extensions import Literal
from ..._models import BaseModel
from ..completion_usage import CompletionUsage
from .chat_completion_message import ChatCompletionMessage
from .chat_completion_token_logprob import ChatCompletionTokenLogprob
class ChoiceLogprobs(BaseModel):
    content: Optional[List[ChatCompletionTokenLogprob]] = None
    'A list of message content tokens with log probability information.'