from typing import List, Optional
from typing_extensions import Literal
from ..._models import BaseModel
from ..completion_usage import CompletionUsage
from .chat_completion_message import ChatCompletionMessage
from .chat_completion_token_logprob import ChatCompletionTokenLogprob
class ChatCompletion(BaseModel):
    id: str
    'A unique identifier for the chat completion.'
    choices: List[Choice]
    'A list of chat completion choices.\n\n    Can be more than one if `n` is greater than 1.\n    '
    created: int
    'The Unix timestamp (in seconds) of when the chat completion was created.'
    model: str
    'The model used for the chat completion.'
    object: Literal['chat.completion']
    'The object type, which is always `chat.completion`.'
    system_fingerprint: Optional[str] = None
    'This fingerprint represents the backend configuration that the model runs with.\n\n    Can be used in conjunction with the `seed` request parameter to understand when\n    backend changes have been made that might impact determinism.\n    '
    usage: Optional[CompletionUsage] = None
    'Usage statistics for the completion request.'