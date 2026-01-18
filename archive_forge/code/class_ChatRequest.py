import time
from dataclasses import asdict, dataclass, field
from typing import List, Literal, Optional
from mlflow.types.schema import Array, ColSpec, DataType, Object, Property, Schema
@dataclass()
class ChatRequest(ChatParams):
    """
    Format of the request object expected by the chat endpoint.

    Args:
        messages (List[:py:class:`ChatMessage`]): A list of :py:class:`ChatMessage`
            that will be passed to the model. **Optional**, defaults to empty list (``[]``)
        temperature (float): A param used to control randomness and creativity during inference.
            **Optional**, defaults to ``1.0``
        max_tokens (int): The maximum number of new tokens to generate.
            **Optional**, defaults to ``None`` (unlimited)
        stop (List[str]): A list of tokens at which to stop generation. **Optional**,
            defaults to ``None``
        n (int): The number of responses to generate. **Optional**,
            defaults to ``1``
        stream (bool): Whether to stream back responses as they are generated. **Optional**,
            defaults to ``False``
    """
    messages: List[ChatMessage] = field(default_factory=list)

    def __post_init__(self):
        self._convert_dataclass_list('messages', ChatMessage)
        super().__post_init__()