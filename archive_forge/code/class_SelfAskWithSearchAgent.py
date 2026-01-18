from typing import Any, Sequence, Union
from langchain_community.utilities.google_serper import GoogleSerperAPIWrapper
from langchain_community.utilities.searchapi import SearchApiAPIWrapper
from langchain_community.utilities.serpapi import SerpAPIWrapper
from langchain_core._api import deprecated
from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts import BasePromptTemplate
from langchain_core.pydantic_v1 import Field
from langchain_core.runnables import Runnable, RunnablePassthrough
from langchain_core.tools import BaseTool, Tool
from langchain.agents.agent import Agent, AgentExecutor, AgentOutputParser
from langchain.agents.agent_types import AgentType
from langchain.agents.format_scratchpad import format_log_to_str
from langchain.agents.self_ask_with_search.output_parser import SelfAskOutputParser
from langchain.agents.self_ask_with_search.prompt import PROMPT
from langchain.agents.utils import validate_tools_single_input
@deprecated('0.1.0', alternative='create_self_ask_with_search', removal='0.2.0')
class SelfAskWithSearchAgent(Agent):
    """Agent for the self-ask-with-search paper."""
    output_parser: AgentOutputParser = Field(default_factory=SelfAskOutputParser)

    @classmethod
    def _get_default_output_parser(cls, **kwargs: Any) -> AgentOutputParser:
        return SelfAskOutputParser()

    @property
    def _agent_type(self) -> str:
        """Return Identifier of an agent type."""
        return AgentType.SELF_ASK_WITH_SEARCH

    @classmethod
    def create_prompt(cls, tools: Sequence[BaseTool]) -> BasePromptTemplate:
        """Prompt does not depend on tools."""
        return PROMPT

    @classmethod
    def _validate_tools(cls, tools: Sequence[BaseTool]) -> None:
        validate_tools_single_input(cls.__name__, tools)
        super()._validate_tools(tools)
        if len(tools) != 1:
            raise ValueError(f'Exactly one tool must be specified, but got {tools}')
        tool_names = {tool.name for tool in tools}
        if tool_names != {'Intermediate Answer'}:
            raise ValueError(f'Tool name should be Intermediate Answer, got {tool_names}')

    @property
    def observation_prefix(self) -> str:
        """Prefix to append the observation with."""
        return 'Intermediate answer: '

    @property
    def llm_prefix(self) -> str:
        """Prefix to append the LLM call with."""
        return ''