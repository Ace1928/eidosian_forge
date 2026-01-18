from typing import List, Optional
from typing_extensions import Literal
from ..._models import BaseModel
from ..completion_usage import CompletionUsage
from .chat_completion_message import ChatCompletionMessage
from .chat_completion_token_logprob import ChatCompletionTokenLogprob
Usage statistics for the completion request.