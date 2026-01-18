import time
from dataclasses import asdict, dataclass, field
from typing import List, Literal, Optional
from mlflow.types.schema import Array, ColSpec, DataType, Object, Property, Schema
@dataclass
class ChatResponse(_BaseDataclass):
    """
    The full response object returned by the chat endpoint.

    Args:
        id (str): The ID of the response.
        object (str): The object type.
        created (int): The time the response was created.
            **Optional**, defaults to the current time.
        model (str): The name of the model used.
        choices (List[:py:class:`ChatChoice`]): A list of :py:class:`ChatChoice` objects
            containing the generated responses
        usage (:py:class:`TokenUsageStats`): An object describing the tokens used by the request.
    """
    id: str
    model: str
    choices: List[ChatChoice]
    usage: TokenUsageStats
    object: Literal['chat.completion'] = 'chat.completion'
    created: int = field(default_factory=lambda: int(time.time()))

    def __post_init__(self):
        self._validate_field('id', str, True)
        self._validate_field('object', str, True)
        self._validate_field('created', int, True)
        self._validate_field('model', str, True)
        self._convert_dataclass_list('choices', ChatChoice)
        if isinstance(self.usage, dict):
            self.usage = TokenUsageStats(**self.usage)
        if not isinstance(self.usage, TokenUsageStats):
            raise ValueError(f'Expected `usage` to be of type TokenUsageStats or dict, got {type(self.usage)}')