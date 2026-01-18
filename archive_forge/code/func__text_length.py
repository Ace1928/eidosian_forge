from pathlib import Path
from typing import Any, Dict, List
from langchain_core.embeddings import Embeddings
from langchain_core.pydantic_v1 import BaseModel, Extra, Field
def _text_length(self, text: Any) -> int:
    """
        Help function to get the length for the input text. Text can be either
        a list of ints (which means a single text as input), or a tuple of list of ints
        (representing several text inputs to the model).
        """
    if isinstance(text, dict):
        return len(next(iter(text.values())))
    elif not hasattr(text, '__len__'):
        return 1
    elif len(text) == 0 or isinstance(text[0], int):
        return len(text)
    else:
        return sum([len(t) for t in text])