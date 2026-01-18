import warnings
from dataclasses import field
from enum import Enum
from typing import List, NoReturn, Optional
from requests import HTTPError
from ..utils import is_pydantic_available
@dataclass
class Details:
    """
    Represents details of a text generation.

    Args:
        finish_reason (`FinishReason`):
            The reason for the generation to finish, represented by a `FinishReason` value.
        generated_tokens (`int`):
            The number of generated tokens.
        seed (`Optional[int]`):
            The sampling seed if sampling was activated.
        prefill (`List[InputToken]`, *optional*):
            The decoder input tokens. Empty if `decoder_input_details` is False. Defaults to an empty list.
        tokens (`List[Token]`):
            The generated tokens. Defaults to an empty list.
        best_of_sequences (`Optional[List[BestOfSequence]]`):
            Additional sequences when using the `best_of` parameter.
    """
    finish_reason: FinishReason
    generated_tokens: int
    seed: Optional[int] = None
    prefill: List[InputToken] = field(default_factory=lambda: [])
    tokens: List[Token] = field(default_factory=lambda: [])
    best_of_sequences: Optional[List[BestOfSequence]] = None

    def __post_init__(self):
        if not is_pydantic_available():
            self.prefill = [InputToken(**input_token) if isinstance(input_token, dict) else input_token for input_token in self.prefill]
            self.tokens = [Token(**token) if isinstance(token, dict) else token for token in self.tokens]
            if self.best_of_sequences is not None:
                self.best_of_sequences = [BestOfSequence(**best_of_sequence) if isinstance(best_of_sequence, dict) else best_of_sequence for best_of_sequence in self.best_of_sequences]