import os
from langchain_community.agent_toolkits import ZapierToolkit
from langchain_community.utilities.zapier import ZapierNLAWrapper
from typing import Any, Dict, Optional
from langchain_core._api import warn_deprecated
from langchain_core.callbacks import (
from langchain_core.pydantic_v1 import Field, root_validator
from langchain_core.tools import BaseTool
from langchain_community.tools.zapier.prompt import BASE_ZAPIER_TOOL_PROMPT
from langchain_community.utilities.zapier import ZapierNLAWrapper
class ZapierNLAListActions(BaseTool):
    """Tool to list all exposed actions for the user."""
    name: str = 'ZapierNLA_list_actions'
    description: str = BASE_ZAPIER_TOOL_PROMPT + "This tool returns a list of the user's exposed actions."
    api_wrapper: ZapierNLAWrapper = Field(default_factory=ZapierNLAWrapper)

    def _run(self, _: str='', run_manager: Optional[CallbackManagerForToolRun]=None) -> str:
        """Use the Zapier NLA tool to return a list of all exposed user actions."""
        warn_deprecated(since='0.0.319', message='This tool will be deprecated on 2023-11-17. See https://nla.zapier.com/sunset/ for details')
        return self.api_wrapper.list_as_str()

    async def _arun(self, _: str='', run_manager: Optional[AsyncCallbackManagerForToolRun]=None) -> str:
        """Use the Zapier NLA tool to return a list of all exposed user actions."""
        warn_deprecated(since='0.0.319', message='This tool will be deprecated on 2023-11-17. See https://nla.zapier.com/sunset/ for details')
        return await self.api_wrapper.alist_as_str()