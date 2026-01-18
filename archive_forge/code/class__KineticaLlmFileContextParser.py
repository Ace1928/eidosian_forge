import json
import logging
import os
import re
from importlib.metadata import version
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, cast
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
from langchain_core.output_parsers.transform import BaseOutputParser
from langchain_core.outputs import ChatGeneration, ChatResult, Generation
from langchain_core.pydantic_v1 import BaseModel, Field, root_validator
class _KineticaLlmFileContextParser:
    """Parser for Kinetica LLM context datafiles."""
    PARSER = re.compile('^<\\|(?P<role>\\w+)\\|>\\W*(?P<content>.*)$', re.DOTALL)

    @classmethod
    def _removesuffix(cls, text: str, suffix: str) -> str:
        if suffix and text.endswith(suffix):
            return text[:-len(suffix)]
        return text

    @classmethod
    def parse_dialogue_file(cls, input_file: os.PathLike) -> Dict:
        path = Path(input_file)
        schema = cls._removesuffix(path.name, '.txt')
        lines = open(input_file).read()
        return cls.parse_dialogue(lines, schema)

    @classmethod
    def parse_dialogue(cls, text: str, schema: str) -> Dict:
        messages = []
        system = None
        lines = text.split('<|end|>')
        user_message = None
        for idx, line in enumerate(lines):
            line = line.strip()
            if len(line) == 0:
                continue
            match = cls.PARSER.match(line)
            if match is None:
                raise ValueError(f'Could not find starting token in: {line}')
            groupdict = match.groupdict()
            role = groupdict['role']
            if role == 'system':
                if system is not None:
                    raise ValueError(f'Only one system token allowed in: {line}')
                system = groupdict['content']
            elif role == 'user':
                if user_message is not None:
                    raise ValueError(f'Found user token without assistant token: {line}')
                user_message = groupdict
            elif role == 'assistant':
                if user_message is None:
                    raise Exception(f'Found assistant token without user token: {line}')
                messages.append(user_message)
                messages.append(groupdict)
                user_message = None
            else:
                raise ValueError(f'Unknown token: {role}')
        return {'schema': schema, 'system': system, 'messages': messages}