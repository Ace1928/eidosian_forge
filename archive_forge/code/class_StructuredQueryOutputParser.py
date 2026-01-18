from __future__ import annotations
import json
from typing import Any, Callable, List, Optional, Sequence, Tuple, Union, cast
from langchain_core.exceptions import OutputParserException
from langchain_core.language_models import BaseLanguageModel
from langchain_core.output_parsers import BaseOutputParser
from langchain_core.output_parsers.json import parse_and_check_json_markdown
from langchain_core.prompts import BasePromptTemplate
from langchain_core.prompts.few_shot import FewShotPromptTemplate
from langchain_core.runnables import Runnable
from langchain.chains.llm import LLMChain
from langchain.chains.query_constructor.ir import (
from langchain.chains.query_constructor.parser import get_parser
from langchain.chains.query_constructor.prompt import (
from langchain.chains.query_constructor.schema import AttributeInfo
class StructuredQueryOutputParser(BaseOutputParser[StructuredQuery]):
    """Output parser that parses a structured query."""
    ast_parse: Callable
    'Callable that parses dict into internal representation of query language.'

    def parse(self, text: str) -> StructuredQuery:
        try:
            expected_keys = ['query', 'filter']
            allowed_keys = ['query', 'filter', 'limit']
            parsed = parse_and_check_json_markdown(text, expected_keys)
            if parsed['query'] is None or len(parsed['query']) == 0:
                parsed['query'] = ' '
            if parsed['filter'] == 'NO_FILTER' or not parsed['filter']:
                parsed['filter'] = None
            else:
                parsed['filter'] = self.ast_parse(parsed['filter'])
            if not parsed.get('limit'):
                parsed.pop('limit', None)
            return StructuredQuery(**{k: v for k, v in parsed.items() if k in allowed_keys})
        except Exception as e:
            raise OutputParserException(f'Parsing text\n{text}\n raised following error:\n{e}')

    @classmethod
    def from_components(cls, allowed_comparators: Optional[Sequence[Comparator]]=None, allowed_operators: Optional[Sequence[Operator]]=None, allowed_attributes: Optional[Sequence[str]]=None, fix_invalid: bool=False) -> StructuredQueryOutputParser:
        """
        Create a structured query output parser from components.

        Args:
            allowed_comparators: allowed comparators
            allowed_operators: allowed operators

        Returns:
            a structured query output parser
        """
        ast_parse: Callable
        if fix_invalid:

            def ast_parse(raw_filter: str) -> Optional[FilterDirective]:
                filter = cast(Optional[FilterDirective], get_parser().parse(raw_filter))
                fixed = fix_filter_directive(filter, allowed_comparators=allowed_comparators, allowed_operators=allowed_operators, allowed_attributes=allowed_attributes)
                return fixed
        else:
            ast_parse = get_parser(allowed_comparators=allowed_comparators, allowed_operators=allowed_operators, allowed_attributes=allowed_attributes).parse
        return cls(ast_parse=ast_parse)