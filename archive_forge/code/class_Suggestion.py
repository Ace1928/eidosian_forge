from typing import Optional
from ._util import to_string
class Suggestion:
    """
    Represents a single suggestion being sent or returned from the
    autocomplete server
    """

    def __init__(self, string: str, score: float=1.0, payload: Optional[str]=None) -> None:
        self.string = to_string(string)
        self.payload = to_string(payload)
        self.score = score

    def __repr__(self) -> str:
        return self.string