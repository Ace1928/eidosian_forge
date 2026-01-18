from __future__ import annotations
from typing import Any, Dict, List, Literal, Optional
from langchain_core.load import Serializable
from langchain_core.utils._merge import merge_dicts
class GenerationChunk(Generation):
    """Generation chunk, which can be concatenated with other Generation chunks."""

    @classmethod
    def get_lc_namespace(cls) -> List[str]:
        """Get the namespace of the langchain object."""
        return ['langchain', 'schema', 'output']

    def __add__(self, other: GenerationChunk) -> GenerationChunk:
        if isinstance(other, GenerationChunk):
            generation_info = merge_dicts(self.generation_info or {}, other.generation_info or {})
            return GenerationChunk(text=self.text + other.text, generation_info=generation_info or None)
        else:
            raise TypeError(f"unsupported operand type(s) for +: '{type(self)}' and '{type(other)}'")