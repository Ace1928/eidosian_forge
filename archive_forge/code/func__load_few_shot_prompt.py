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
def _load_few_shot_prompt(config: dict) -> FewShotPromptTemplate:
    """Load the "few shot" prompt from the config."""
    config = _load_template('suffix', config)
    config = _load_template('prefix', config)
    if 'example_prompt_path' in config:
        if 'example_prompt' in config:
            raise ValueError('Only one of example_prompt and example_prompt_path should be specified.')
        config['example_prompt'] = load_prompt(config.pop('example_prompt_path'))
    else:
        config['example_prompt'] = load_prompt_from_config(config['example_prompt'])
    config = _load_examples(config)
    config = _load_output_parser(config)
    return FewShotPromptTemplate(**config)