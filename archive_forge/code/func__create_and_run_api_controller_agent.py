import json
import re
from functools import partial
from typing import Any, Callable, Dict, List, Optional, cast
import yaml
from langchain_core.callbacks import BaseCallbackManager
from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts import BasePromptTemplate, PromptTemplate
from langchain_core.pydantic_v1 import Field
from langchain_core.tools import BaseTool, Tool
from langchain_community.agent_toolkits.openapi.planner_prompt import (
from langchain_community.agent_toolkits.openapi.spec import ReducedOpenAPISpec
from langchain_community.llms import OpenAI
from langchain_community.tools.requests.tool import BaseRequestsTool
from langchain_community.utilities.requests import RequestsWrapper
def _create_and_run_api_controller_agent(plan_str: str) -> str:
    pattern = '\\b(GET|POST|PATCH|DELETE)\\s+(/\\S+)*'
    matches = re.findall(pattern, plan_str)
    endpoint_names = ['{method} {route}'.format(method=method, route=route.split('?')[0]) for method, route in matches]
    docs_str = ''
    for endpoint_name in endpoint_names:
        found_match = False
        for name, _, docs in api_spec.endpoints:
            regex_name = re.compile(re.sub('\\{.*?\\}', '.*', name))
            if regex_name.match(endpoint_name):
                found_match = True
                docs_str += f'== Docs for {endpoint_name} == \n{yaml.dump(docs)}\n'
        if not found_match:
            raise ValueError(f'{endpoint_name} endpoint does not exist.')
    agent = _create_api_controller_agent(base_url, docs_str, requests_wrapper, llm, allow_dangerous_requests)
    return agent.run(plan_str)