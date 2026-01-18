import json
import logging
from pathlib import Path
from typing import Any, List, Optional, Union
import yaml
from langchain_core._api import deprecated
from langchain_core.language_models import BaseLanguageModel
from langchain_core.tools import Tool
from langchain.agents.agent import BaseMultiActionAgent, BaseSingleActionAgent
from langchain.agents.types import AGENT_TO_CLASS
from langchain.chains.loading import load_chain, load_chain_from_config
def _load_agent_from_file(file: Union[str, Path], **kwargs: Any) -> Union[BaseSingleActionAgent, BaseMultiActionAgent]:
    """Load agent from file."""
    valid_suffixes = {'json', 'yaml'}
    if isinstance(file, str):
        file_path = Path(file)
    else:
        file_path = file
    if file_path.suffix[1:] == 'json':
        with open(file_path) as f:
            config = json.load(f)
    elif file_path.suffix[1:] == 'yaml':
        with open(file_path, 'r') as f:
            config = yaml.safe_load(f)
    else:
        raise ValueError(f'Unsupported file type, must be one of {valid_suffixes}.')
    return load_agent_from_config(config, **kwargs)