from __future__ import annotations
import json
import re
from collections import defaultdict
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union
import requests
from langchain_community.chat_models import ChatOpenAI
from langchain_community.utilities.openapi import OpenAPISpec
from langchain_core.callbacks import CallbackManagerForChainRun
from langchain_core.language_models import BaseLanguageModel
from langchain_core.output_parsers.openai_functions import JsonOutputFunctionsParser
from langchain_core.prompts import BasePromptTemplate, ChatPromptTemplate
from langchain_core.utils.input import get_colored_text
from requests import Response
from langchain.chains.base import Chain
from langchain.chains.llm import LLMChain
from langchain.chains.sequential import SequentialChain
from langchain.tools import APIOperation
def _format_url(url: str, path_params: dict) -> str:
    expected_path_param = re.findall('{(.*?)}', url)
    new_params = {}
    for param in expected_path_param:
        clean_param = param.lstrip('.;').rstrip('*')
        val = path_params[clean_param]
        if isinstance(val, list):
            if param[0] == '.':
                sep = '.' if param[-1] == '*' else ','
                new_val = '.' + sep.join(val)
            elif param[0] == ';':
                sep = f'{clean_param}=' if param[-1] == '*' else ','
                new_val = f'{clean_param}=' + sep.join(val)
            else:
                new_val = ','.join(val)
        elif isinstance(val, dict):
            kv_sep = '=' if param[-1] == '*' else ','
            kv_strs = [kv_sep.join((k, v)) for k, v in val.items()]
            if param[0] == '.':
                sep = '.'
                new_val = '.'
            elif param[0] == ';':
                sep = ';'
                new_val = ';'
            else:
                sep = ','
                new_val = ''
            new_val += sep.join(kv_strs)
        elif param[0] == '.':
            new_val = f'.{val}'
        elif param[0] == ';':
            new_val = f';{clean_param}={val}'
        else:
            new_val = val
        new_params[param] = new_val
    return url.format(**new_params)