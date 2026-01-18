from typing import Any, List, Optional, Sequence, Tuple
from langchain_core._api import deprecated
from langchain_core.agents import AgentAction
from langchain_core.callbacks import BaseCallbackManager
from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts import BasePromptTemplate
from langchain_core.prompts.chat import (
from langchain_core.pydantic_v1 import Field
from langchain_core.tools import BaseTool
from langchain.agents.agent import Agent, AgentOutputParser
from langchain.agents.chat.output_parser import ChatOutputParser
from langchain.agents.chat.prompt import (
from langchain.agents.utils import validate_tools_single_input
from langchain.chains.llm import LLMChain
def _construct_scratchpad(self, intermediate_steps: List[Tuple[AgentAction, str]]) -> str:
    agent_scratchpad = super()._construct_scratchpad(intermediate_steps)
    if not isinstance(agent_scratchpad, str):
        raise ValueError('agent_scratchpad should be of type string.')
    if agent_scratchpad:
        return f"This was your previous work (but I haven't seen any of it! I only see what you return as final answer):\n{agent_scratchpad}"
    else:
        return agent_scratchpad