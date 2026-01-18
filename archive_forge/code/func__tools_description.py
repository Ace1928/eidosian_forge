import re
from typing import (
from langchain_core.agents import AgentAction
from langchain_core.exceptions import OutputParserException
from langchain_core.language_models import BaseLanguageModel
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.output_parsers import BaseOutputParser
from langchain_core.pydantic_v1 import Extra, Field
from langchain_core.tools import BaseTool
from langchain.callbacks.manager import (
from langchain.chains.llm import LLMChain
from langchain.evaluation.agents.trajectory_eval_prompt import (
from langchain.evaluation.schema import AgentTrajectoryEvaluator, LLMEvalChain
@property
def _tools_description(self) -> str:
    """Get the description of the agent tools.

        Returns:
            str: The description of the agent tools.
        """
    if self.agent_tools is None:
        return ''
    return '\n\n'.join([f'Tool {i}: {tool.name}\nDescription: {tool.description}' for i, tool in enumerate(self.agent_tools, 1)])