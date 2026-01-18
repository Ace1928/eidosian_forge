from __future__ import annotations
import asyncio
import json
import logging
import time
from abc import abstractmethod
from pathlib import Path
from typing import (
import yaml
from langchain_core._api import deprecated
from langchain_core.agents import AgentAction, AgentFinish, AgentStep
from langchain_core.callbacks import (
from langchain_core.exceptions import OutputParserException
from langchain_core.language_models import BaseLanguageModel
from langchain_core.messages import BaseMessage
from langchain_core.output_parsers import BaseOutputParser
from langchain_core.prompts import BasePromptTemplate
from langchain_core.prompts.few_shot import FewShotPromptTemplate
from langchain_core.prompts.prompt import PromptTemplate
from langchain_core.pydantic_v1 import BaseModel, root_validator
from langchain_core.runnables import Runnable, RunnableConfig, ensure_config
from langchain_core.runnables.utils import AddableDict
from langchain_core.tools import BaseTool
from langchain_core.utils.input import get_color_mapping
from langchain.agents.agent_iterator import AgentExecutorIterator
from langchain.agents.agent_types import AgentType
from langchain.agents.tools import InvalidTool
from langchain.chains.base import Chain
from langchain.chains.llm import LLMChain
from langchain.utilities.asyncio import asyncio_timeout
def _get_tool_return(self, next_step_output: Tuple[AgentAction, str]) -> Optional[AgentFinish]:
    """Check if the tool is a returning tool."""
    agent_action, observation = next_step_output
    name_to_tool_map = {tool.name: tool for tool in self.tools}
    return_value_key = 'output'
    if len(self.agent.return_values) > 0:
        return_value_key = self.agent.return_values[0]
    if agent_action.tool in name_to_tool_map:
        if name_to_tool_map[agent_action.tool].return_direct:
            return AgentFinish({return_value_key: observation}, '')
    return None