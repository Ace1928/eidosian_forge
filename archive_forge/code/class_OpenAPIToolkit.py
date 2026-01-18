from __future__ import annotations
from typing import Any, List
from langchain_core.language_models import BaseLanguageModel
from langchain_core.tools import Tool
from langchain_community.agent_toolkits.base import BaseToolkit
from langchain_community.agent_toolkits.json.base import create_json_agent
from langchain_community.agent_toolkits.json.toolkit import JsonToolkit
from langchain_community.agent_toolkits.openapi.prompt import DESCRIPTION
from langchain_community.tools import BaseTool
from langchain_community.tools.json.tool import JsonSpec
from langchain_community.tools.requests.tool import (
from langchain_community.utilities.requests import TextRequestsWrapper
class OpenAPIToolkit(BaseToolkit):
    """Toolkit for interacting with an OpenAPI API.

    *Security Note*: This toolkit contains tools that can read and modify
        the state of a service; e.g., by creating, deleting, or updating,
        reading underlying data.

        For example, this toolkit can be used to delete data exposed via
        an OpenAPI compliant API.
    """
    json_agent: Any
    requests_wrapper: TextRequestsWrapper
    allow_dangerous_requests: bool = False
    'Allow dangerous requests. See documentation for details.'

    def get_tools(self) -> List[BaseTool]:
        """Get the tools in the toolkit."""
        json_agent_tool = Tool(name='json_explorer', func=self.json_agent.run, description=DESCRIPTION)
        request_toolkit = RequestsToolkit(requests_wrapper=self.requests_wrapper, allow_dangerous_requests=self.allow_dangerous_requests)
        return [*request_toolkit.get_tools(), json_agent_tool]

    @classmethod
    def from_llm(cls, llm: BaseLanguageModel, json_spec: JsonSpec, requests_wrapper: TextRequestsWrapper, allow_dangerous_requests: bool=False, **kwargs: Any) -> OpenAPIToolkit:
        """Create json agent from llm, then initialize."""
        json_agent = create_json_agent(llm, JsonToolkit(spec=json_spec), **kwargs)
        return cls(json_agent=json_agent, requests_wrapper=requests_wrapper, allow_dangerous_requests=allow_dangerous_requests)