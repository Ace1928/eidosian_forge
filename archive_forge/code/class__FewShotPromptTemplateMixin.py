from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union
from langchain_core.example_selectors import BaseExampleSelector
from langchain_core.messages import BaseMessage, get_buffer_string
from langchain_core.prompts.chat import (
from langchain_core.prompts.prompt import PromptTemplate
from langchain_core.prompts.string import (
from langchain_core.pydantic_v1 import BaseModel, Extra, Field, root_validator
class _FewShotPromptTemplateMixin(BaseModel):
    """Prompt template that contains few shot examples."""
    examples: Optional[List[dict]] = None
    'Examples to format into the prompt.\n    Either this or example_selector should be provided.'
    example_selector: Optional[BaseExampleSelector] = None
    'ExampleSelector to choose the examples to format into the prompt.\n    Either this or examples should be provided.'

    class Config:
        """Configuration for this pydantic object."""
        extra = Extra.forbid
        arbitrary_types_allowed = True

    @root_validator(pre=True)
    def check_examples_and_selector(cls, values: Dict) -> Dict:
        """Check that one and only one of examples/example_selector are provided."""
        examples = values.get('examples', None)
        example_selector = values.get('example_selector', None)
        if examples and example_selector:
            raise ValueError("Only one of 'examples' and 'example_selector' should be provided")
        if examples is None and example_selector is None:
            raise ValueError("One of 'examples' and 'example_selector' should be provided")
        return values

    def _get_examples(self, **kwargs: Any) -> List[dict]:
        """Get the examples to use for formatting the prompt.

        Args:
            **kwargs: Keyword arguments to be passed to the example selector.

        Returns:
            List of examples.
        """
        if self.examples is not None:
            return self.examples
        elif self.example_selector is not None:
            return self.example_selector.select_examples(kwargs)
        else:
            raise ValueError("One of 'examples' and 'example_selector' should be provided")

    async def _aget_examples(self, **kwargs: Any) -> List[dict]:
        """Get the examples to use for formatting the prompt.

        Args:
            **kwargs: Keyword arguments to be passed to the example selector.

        Returns:
            List of examples.
        """
        if self.examples is not None:
            return self.examples
        elif self.example_selector is not None:
            return await self.example_selector.aselect_examples(kwargs)
        else:
            raise ValueError("One of 'examples' and 'example_selector' should be provided")