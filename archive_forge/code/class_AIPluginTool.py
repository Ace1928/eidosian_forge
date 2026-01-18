from __future__ import annotations
import json
from typing import Optional, Type
import requests
import yaml
from langchain_core.callbacks import (
from langchain_core.pydantic_v1 import BaseModel
from langchain_core.tools import BaseTool
class AIPluginTool(BaseTool):
    """Tool for getting the OpenAPI spec for an AI Plugin."""
    plugin: AIPlugin
    api_spec: str
    args_schema: Type[AIPluginToolSchema] = AIPluginToolSchema

    @classmethod
    def from_plugin_url(cls, url: str) -> AIPluginTool:
        plugin = AIPlugin.from_url(url)
        description = f'Call this tool to get the OpenAPI spec (and usage guide) for interacting with the {plugin.name_for_human} API. You should only call this ONCE! What is the {plugin.name_for_human} API useful for? ' + plugin.description_for_human
        open_api_spec_str = requests.get(plugin.api.url).text
        open_api_spec = marshal_spec(open_api_spec_str)
        api_spec = f'Usage Guide: {plugin.description_for_model}\n\nOpenAPI Spec: {open_api_spec}'
        return cls(name=plugin.name_for_model, description=description, plugin=plugin, api_spec=api_spec)

    def _run(self, tool_input: Optional[str]='', run_manager: Optional[CallbackManagerForToolRun]=None) -> str:
        """Use the tool."""
        return self.api_spec

    async def _arun(self, tool_input: Optional[str]=None, run_manager: Optional[AsyncCallbackManagerForToolRun]=None) -> str:
        """Use the tool asynchronously."""
        return self.api_spec