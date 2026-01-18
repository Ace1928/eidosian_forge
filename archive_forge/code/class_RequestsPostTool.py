import json
from typing import Any, Dict, Optional, Union
from langchain_core.pydantic_v1 import BaseModel
from langchain_core.callbacks import (
from langchain_community.utilities.requests import GenericRequestsWrapper
from langchain_core.tools import BaseTool
class RequestsPostTool(BaseRequestsTool, BaseTool):
    """Tool for making a POST request to an API endpoint."""
    name: str = 'requests_post'
    description: str = 'Use this when you want to POST to a website.\n    Input should be a json string with two keys: "url" and "data".\n    The value of "url" should be a string, and the value of "data" should be a dictionary of \n    key-value pairs you want to POST to the url.\n    Be careful to always use double quotes for strings in the json string\n    The output will be the text response of the POST request.\n    '

    def _run(self, text: str, run_manager: Optional[CallbackManagerForToolRun]=None) -> Union[str, Dict[str, Any]]:
        """Run the tool."""
        try:
            data = _parse_input(text)
            return self.requests_wrapper.post(_clean_url(data['url']), data['data'])
        except Exception as e:
            return repr(e)

    async def _arun(self, text: str, run_manager: Optional[AsyncCallbackManagerForToolRun]=None) -> Union[str, Dict[str, Any]]:
        """Run the tool asynchronously."""
        try:
            data = _parse_input(text)
            return await self.requests_wrapper.apost(_clean_url(data['url']), data['data'])
        except Exception as e:
            return repr(e)