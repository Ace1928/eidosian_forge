from __future__ import annotations
from typing import Any, Callable, List, NamedTuple, Optional, Sequence
from langchain_core._api import deprecated
from langchain_core.callbacks import BaseCallbackManager
from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts import PromptTemplate
from langchain_core.pydantic_v1 import Field
from langchain_core.tools import BaseTool
from langchain.agents.agent import Agent, AgentExecutor, AgentOutputParser
from langchain.agents.agent_types import AgentType
from langchain.agents.mrkl.output_parser import MRKLOutputParser
from langchain.agents.mrkl.prompt import FORMAT_INSTRUCTIONS, PREFIX, SUFFIX
from langchain.agents.tools import Tool
from langchain.agents.utils import validate_tools_single_input
from langchain.chains import LLMChain
from langchain.tools.render import render_text_description
@deprecated('0.1.0', removal='0.2.0')
class MRKLChain(AgentExecutor):
    """[Deprecated] Chain that implements the MRKL system."""

    @classmethod
    def from_chains(cls, llm: BaseLanguageModel, chains: List[ChainConfig], **kwargs: Any) -> AgentExecutor:
        """User friendly way to initialize the MRKL chain.

        This is intended to be an easy way to get up and running with the
        MRKL chain.

        Args:
            llm: The LLM to use as the agent LLM.
            chains: The chains the MRKL system has access to.
            **kwargs: parameters to be passed to initialization.

        Returns:
            An initialized MRKL chain.
        """
        tools = [Tool(name=c.action_name, func=c.action, description=c.action_description) for c in chains]
        agent = ZeroShotAgent.from_llm_and_tools(llm, tools)
        return cls(agent=agent, tools=tools, **kwargs)