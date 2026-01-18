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
def _load_examples(config: dict) -> dict:
    """Load examples if necessary."""
    if isinstance(config['examples'], list):
        pass
    elif isinstance(config['examples'], str):
        with open(config['examples']) as f:
            if config['examples'].endswith('.json'):
                examples = json.load(f)
            elif config['examples'].endswith(('.yaml', '.yml')):
                examples = yaml.safe_load(f)
            else:
                raise ValueError('Invalid file format. Only json or yaml formats are supported.')
        config['examples'] = examples
    else:
        raise ValueError('Invalid examples format. Only list or string are supported.')
    return config