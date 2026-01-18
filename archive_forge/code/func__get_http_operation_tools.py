from __future__ import annotations
from typing import Any, List, Optional, Sequence
from langchain_core.language_models import BaseLanguageModel
from langchain_core.pydantic_v1 import Field
from langchain_core.tools import BaseTool
from langchain_community.agent_toolkits.base import BaseToolkit
from langchain_community.agent_toolkits.nla.tool import NLATool
from langchain_community.tools.openapi.utils.openapi_utils import OpenAPISpec
from langchain_community.tools.plugin import AIPlugin
from langchain_community.utilities.requests import Requests
@staticmethod
def _get_http_operation_tools(llm: BaseLanguageModel, spec: OpenAPISpec, requests: Optional[Requests]=None, verbose: bool=False, **kwargs: Any) -> List[NLATool]:
    """Get the tools for all the API operations."""
    if not spec.paths:
        return []
    http_operation_tools = []
    for path in spec.paths:
        for method in spec.get_methods_for_path(path):
            endpoint_tool = NLATool.from_llm_and_method(llm=llm, path=path, method=method, spec=spec, requests=requests, verbose=verbose, **kwargs)
            http_operation_tools.append(endpoint_tool)
    return http_operation_tools