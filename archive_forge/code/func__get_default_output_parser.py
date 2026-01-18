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
@classmethod
def _get_default_output_parser(cls, **kwargs: Any) -> AgentOutputParser:
    return ChatOutputParser()