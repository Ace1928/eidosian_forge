import json
from json import JSONDecodeError
from typing import List, Union
from langchain_core.agents import AgentAction, AgentActionMessageLog, AgentFinish
from langchain_core.exceptions import OutputParserException
from langchain_core.messages import (
from langchain_core.outputs import ChatGeneration, Generation
from langchain.agents.agent import MultiActionAgentOutputParser
class ToolAgentAction(AgentActionMessageLog):
    tool_call_id: str
    'Tool call that this message is responding to.'