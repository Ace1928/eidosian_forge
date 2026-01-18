import time
from dataclasses import asdict, dataclass, field
from typing import List, Literal, Optional
from mlflow.types.schema import Array, ColSpec, DataType, Object, Property, Schema
@dataclass
class TokenUsageStats(_BaseDataclass):
    """
    Stats about the number of tokens used during inference.

    Args:
        prompt_tokens (int): The number of tokens in the prompt.
        completion_tokens (int): The number of tokens in the generated completion.
        total_tokens (int): The total number of tokens used.
    """
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

    def __post_init__(self):
        self._validate_field('prompt_tokens', int, True)
        self._validate_field('completion_tokens', int, True)
        self._validate_field('total_tokens', int, True)