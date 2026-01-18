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
@deprecated('0.1.0', removal='0.2.0')
class SelfAskWithSearchChain(AgentExecutor):
    """[Deprecated] Chain that does self-ask with search."""

    def __init__(self, llm: BaseLanguageModel, search_chain: Union[GoogleSerperAPIWrapper, SearchApiAPIWrapper, SerpAPIWrapper], **kwargs: Any):
        """Initialize only with an LLM and a search chain."""
        search_tool = Tool(name='Intermediate Answer', func=search_chain.run, coroutine=search_chain.arun, description='Search')
        agent = SelfAskWithSearchAgent.from_llm_and_tools(llm, [search_tool])
        super().__init__(agent=agent, tools=[search_tool], **kwargs)