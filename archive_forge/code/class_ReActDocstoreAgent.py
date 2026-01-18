from typing import Any, List, Optional, Sequence
from langchain_core._api import deprecated
from langchain_core.documents import Document
from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts import BasePromptTemplate
from langchain_core.pydantic_v1 import Field
from langchain_core.tools import BaseTool, Tool
from langchain.agents.agent import Agent, AgentExecutor, AgentOutputParser
from langchain.agents.agent_types import AgentType
from langchain.agents.react.output_parser import ReActOutputParser
from langchain.agents.react.textworld_prompt import TEXTWORLD_PROMPT
from langchain.agents.react.wiki_prompt import WIKI_PROMPT
from langchain.agents.utils import validate_tools_single_input
from langchain.docstore.base import Docstore
@deprecated('0.1.0', removal='0.2.0')
class ReActDocstoreAgent(Agent):
    """Agent for the ReAct chain."""
    output_parser: AgentOutputParser = Field(default_factory=ReActOutputParser)

    @classmethod
    def _get_default_output_parser(cls, **kwargs: Any) -> AgentOutputParser:
        return ReActOutputParser()

    @property
    def _agent_type(self) -> str:
        """Return Identifier of an agent type."""
        return AgentType.REACT_DOCSTORE

    @classmethod
    def create_prompt(cls, tools: Sequence[BaseTool]) -> BasePromptTemplate:
        """Return default prompt."""
        return WIKI_PROMPT

    @classmethod
    def _validate_tools(cls, tools: Sequence[BaseTool]) -> None:
        validate_tools_single_input(cls.__name__, tools)
        super()._validate_tools(tools)
        if len(tools) != 2:
            raise ValueError(f'Exactly two tools must be specified, but got {tools}')
        tool_names = {tool.name for tool in tools}
        if tool_names != {'Lookup', 'Search'}:
            raise ValueError(f'Tool names should be Lookup and Search, got {tool_names}')

    @property
    def observation_prefix(self) -> str:
        """Prefix to append the observation with."""
        return 'Observation: '

    @property
    def _stop(self) -> List[str]:
        return ['\nObservation:']

    @property
    def llm_prefix(self) -> str:
        """Prefix to append the LLM call with."""
        return 'Thought:'