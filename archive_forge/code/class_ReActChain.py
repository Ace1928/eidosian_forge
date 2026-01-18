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
class ReActChain(AgentExecutor):
    """[Deprecated] Chain that implements the ReAct paper."""

    def __init__(self, llm: BaseLanguageModel, docstore: Docstore, **kwargs: Any):
        """Initialize with the LLM and a docstore."""
        docstore_explorer = DocstoreExplorer(docstore)
        tools = [Tool(name='Search', func=docstore_explorer.search, description='Search for a term in the docstore.'), Tool(name='Lookup', func=docstore_explorer.lookup, description='Lookup a term in the docstore.')]
        agent = ReActDocstoreAgent.from_llm_and_tools(llm, tools)
        super().__init__(agent=agent, tools=tools, **kwargs)