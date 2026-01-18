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
class ZapierNLARunAction(BaseTool):
    """Tool to run a specific action from the user's exposed actions.

    Params:
        action_id: a specific action ID (from list actions) of the action to execute
            (the set api_key must be associated with the action owner)
        instructions: a natural language instruction string for using the action
            (eg. "get the latest email from Mike Knoop" for "Gmail: find email" action)
        params: a dict, optional. Any params provided will *override* AI guesses
            from `instructions` (see "understanding the AI guessing flow" here:
            https://nla.zapier.com/docs/using-the-api#ai-guessing)

    """
    api_wrapper: ZapierNLAWrapper = Field(default_factory=ZapierNLAWrapper)
    action_id: str
    params: Optional[dict] = None
    base_prompt: str = BASE_ZAPIER_TOOL_PROMPT
    zapier_description: str
    params_schema: Dict[str, str] = Field(default_factory=dict)
    name: str = ''
    description: str = ''

    @root_validator
    def set_name_description(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        zapier_description = values['zapier_description']
        params_schema = values['params_schema']
        if 'instructions' in params_schema:
            del params_schema['instructions']
        necessary_fields = {'{zapier_description}', '{params}'}
        if not all((field in values['base_prompt'] for field in necessary_fields)):
            raise ValueError('Your custom base Zapier prompt must contain input fields for {zapier_description} and {params}.')
        values['name'] = zapier_description
        values['description'] = values['base_prompt'].format(zapier_description=zapier_description, params=str(list(params_schema.keys())))
        return values

    def _run(self, instructions: str, run_manager: Optional[CallbackManagerForToolRun]=None) -> str:
        """Use the Zapier NLA tool to return a list of all exposed user actions."""
        warn_deprecated(since='0.0.319', message='This tool will be deprecated on 2023-11-17. See https://nla.zapier.com/sunset/ for details')
        return self.api_wrapper.run_as_str(self.action_id, instructions, self.params)

    async def _arun(self, instructions: str, run_manager: Optional[AsyncCallbackManagerForToolRun]=None) -> str:
        """Use the Zapier NLA tool to return a list of all exposed user actions."""
        warn_deprecated(since='0.0.319', message='This tool will be deprecated on 2023-11-17. See https://nla.zapier.com/sunset/ for details')
        return await self.api_wrapper.arun_as_str(self.action_id, instructions, self.params)