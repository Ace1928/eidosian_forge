from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union
from langchain_core.example_selectors import BaseExampleSelector
from langchain_core.messages import BaseMessage, get_buffer_string
from langchain_core.prompts.chat import (
from langchain_core.prompts.prompt import PromptTemplate
from langchain_core.prompts.string import (
from langchain_core.pydantic_v1 import BaseModel, Extra, Field, root_validator
class FewShotPromptTemplate(_FewShotPromptTemplateMixin, StringPromptTemplate):
    """Prompt template that contains few shot examples."""

    @classmethod
    def is_lc_serializable(cls) -> bool:
        """Return whether or not the class is serializable."""
        return False
    validate_template: bool = False
    'Whether or not to try validating the template.'
    input_variables: List[str]
    'A list of the names of the variables the prompt template expects.'
    example_prompt: PromptTemplate
    'PromptTemplate used to format an individual example.'
    suffix: str
    'A prompt template string to put after the examples.'
    example_separator: str = '\n\n'
    'String separator used to join the prefix, the examples, and suffix.'
    prefix: str = ''
    'A prompt template string to put before the examples.'
    template_format: Literal['f-string', 'jinja2'] = 'f-string'
    "The format of the prompt template. Options are: 'f-string', 'jinja2'."

    @root_validator()
    def template_is_valid(cls, values: Dict) -> Dict:
        """Check that prefix, suffix, and input variables are consistent."""
        if values['validate_template']:
            check_valid_template(values['prefix'] + values['suffix'], values['template_format'], values['input_variables'] + list(values['partial_variables']))
        elif values.get('template_format'):
            values['input_variables'] = [var for var in get_template_variables(values['prefix'] + values['suffix'], values['template_format']) if var not in values['partial_variables']]
        return values

    class Config:
        """Configuration for this pydantic object."""
        extra = Extra.forbid
        arbitrary_types_allowed = True

    def format(self, **kwargs: Any) -> str:
        kwargs = self._merge_partial_and_user_variables(**kwargs)
        examples = self._get_examples(**kwargs)
        examples = [{k: e[k] for k in self.example_prompt.input_variables} for e in examples]
        example_strings = [self.example_prompt.format(**example) for example in examples]
        pieces = [self.prefix, *example_strings, self.suffix]
        template = self.example_separator.join([piece for piece in pieces if piece])
        return DEFAULT_FORMATTER_MAPPING[self.template_format](template, **kwargs)

    async def aformat(self, **kwargs: Any) -> str:
        kwargs = self._merge_partial_and_user_variables(**kwargs)
        examples = await self._aget_examples(**kwargs)
        examples = [{k: e[k] for k in self.example_prompt.input_variables} for e in examples]
        example_strings = [await self.example_prompt.aformat(**example) for example in examples]
        pieces = [self.prefix, *example_strings, self.suffix]
        template = self.example_separator.join([piece for piece in pieces if piece])
        return DEFAULT_FORMATTER_MAPPING[self.template_format](template, **kwargs)

    @property
    def _prompt_type(self) -> str:
        """Return the prompt type key."""
        return 'few_shot'

    def save(self, file_path: Union[Path, str]) -> None:
        if self.example_selector:
            raise ValueError('Saving an example selector is not currently supported')
        return super().save(file_path)