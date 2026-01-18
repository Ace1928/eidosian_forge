import json
import logging
from pathlib import Path
from typing import Callable, Dict, Union
import yaml
from langchain_core.output_parsers.string import StrOutputParser
from langchain_core.prompts.base import BasePromptTemplate
from langchain_core.prompts.chat import ChatPromptTemplate
from langchain_core.prompts.few_shot import FewShotPromptTemplate
from langchain_core.prompts.prompt import PromptTemplate
def _load_output_parser(config: dict) -> dict:
    """Load output parser."""
    if 'output_parser' in config and config['output_parser']:
        _config = config.pop('output_parser')
        output_parser_type = _config.pop('_type')
        if output_parser_type == 'default':
            output_parser = StrOutputParser(**_config)
        else:
            raise ValueError(f'Unsupported output parser {output_parser_type}')
        config['output_parser'] = output_parser
    return config