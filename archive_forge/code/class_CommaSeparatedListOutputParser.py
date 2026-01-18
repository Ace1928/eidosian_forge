from __future__ import annotations
import re
from abc import abstractmethod
from collections import deque
from typing import AsyncIterator, Deque, Iterator, List, TypeVar, Union
from langchain_core.messages import BaseMessage
from langchain_core.output_parsers.transform import BaseTransformOutputParser
class CommaSeparatedListOutputParser(ListOutputParser):
    """Parse the output of an LLM call to a comma-separated list."""

    @classmethod
    def is_lc_serializable(cls) -> bool:
        return True

    @classmethod
    def get_lc_namespace(cls) -> List[str]:
        """Get the namespace of the langchain object."""
        return ['langchain', 'output_parsers', 'list']

    def get_format_instructions(self) -> str:
        return 'Your response should be a list of comma separated values, eg: `foo, bar, baz`'

    def parse(self, text: str) -> List[str]:
        """Parse the output of an LLM call."""
        return text.strip().split(', ')

    @property
    def _type(self) -> str:
        return 'comma-separated-list'