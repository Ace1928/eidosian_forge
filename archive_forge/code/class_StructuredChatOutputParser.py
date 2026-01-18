from __future__ import annotations
import json
import logging
import re
from typing import Optional, Union
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.exceptions import OutputParserException
from langchain_core.language_models import BaseLanguageModel
from langchain_core.pydantic_v1 import Field
from langchain.agents.agent import AgentOutputParser
from langchain.agents.structured_chat.prompt import FORMAT_INSTRUCTIONS
from langchain.output_parsers import OutputFixingParser
class StructuredChatOutputParser(AgentOutputParser):
    """Output parser for the structured chat agent."""
    format_instructions: str = FORMAT_INSTRUCTIONS
    'Default formatting instructions'
    pattern = re.compile('```(?:json\\s+)?(\\W.*?)```', re.DOTALL)
    'Regex pattern to parse the output.'

    def get_format_instructions(self) -> str:
        """Returns formatting instructions for the given output parser."""
        return self.format_instructions

    def parse(self, text: str) -> Union[AgentAction, AgentFinish]:
        try:
            action_match = self.pattern.search(text)
            if action_match is not None:
                response = json.loads(action_match.group(1).strip(), strict=False)
                if isinstance(response, list):
                    logger.warning('Got multiple action responses: %s', response)
                    response = response[0]
                if response['action'] == 'Final Answer':
                    return AgentFinish({'output': response['action_input']}, text)
                else:
                    return AgentAction(response['action'], response.get('action_input', {}), text)
            else:
                return AgentFinish({'output': text}, text)
        except Exception as e:
            raise OutputParserException(f'Could not parse LLM output: {text}') from e

    @property
    def _type(self) -> str:
        return 'structured_chat'