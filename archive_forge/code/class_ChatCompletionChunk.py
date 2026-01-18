from typing import List, Optional
from typing_extensions import Literal
from ..._models import BaseModel
from .chat_completion_token_logprob import ChatCompletionTokenLogprob
class ChatCompletionChunk(BaseModel):
    id: str
    'A unique identifier for the chat completion. Each chunk has the same ID.'
    choices: List[Choice]
    'A list of chat completion choices.\n\n    Can be more than one if `n` is greater than 1.\n    '
    created: int
    'The Unix timestamp (in seconds) of when the chat completion was created.\n\n    Each chunk has the same timestamp.\n    '
    model: str
    'The model to generate the completion.'
    object: Literal['chat.completion.chunk']
    'The object type, which is always `chat.completion.chunk`.'
    system_fingerprint: Optional[str] = None
    '\n    This fingerprint represents the backend configuration that the model runs with.\n    Can be used in conjunction with the `seed` request parameter to understand when\n    backend changes have been made that might impact determinism.\n    '