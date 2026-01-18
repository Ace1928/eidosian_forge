from __future__ import annotations
from typing import Any, Dict, Optional
import requests
from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.tools import BaseTool
class CogniswitchKnowledgeSourceURL(BaseTool):
    """Tool that uses the Cogniswitch services to store data from a URL.

    name: str = "cogniswitch_knowledge_source_url"
    description: str = (
        "This calls the CogniSwitch services to analyze & store data from a url.
        the URL is provided in input, assign that value to the url key.
        Assign document name & description only if provided in input"
    )
    """
    name: str = 'cogniswitch_knowledge_source_url'
    description: str = '\n    This calls the CogniSwitch services to analyze & store data from a url. \n        the URL is provided in input, assign that value to the url key. \n        Assign document name & description only if provided in input'
    cs_token: str
    OAI_token: str
    apiKey: str
    knowledgesource_url = 'https://api.cogniswitch.ai:8243/cs-api/0.0.1/cs/knowledgeSource/url'

    def _run(self, url: Optional[str]=None, document_name: Optional[str]=None, document_description: Optional[str]=None, run_manager: Optional[CallbackManagerForToolRun]=None) -> Dict[str, Any]:
        """
        Execute the tool to store the data given from a url.
        This calls the CogniSwitch services to analyze & store data from a url.
        the URL is provided in input, assign that value to the url key.
        Assign document name & description only if provided in input.

        Args:
            url Optional[str]: The website/url link of your knowledge
            document_name Optional[str]: Name of your knowledge document
            document_description Optional[str]: Description of your knowledge document
            run_manager (Optional[CallbackManagerForChainRun]):
            Manager for chain run callbacks.

        Returns:
            Dict[str, Any]: Output dictionary containing
            the 'response' from the service.
        """
        if not url:
            return {'message': 'No input provided'}
        response = self.store_data(url=url, document_name=document_name, document_description=document_description)
        return response

    def store_data(self, url: Optional[str], document_name: Optional[str], document_description: Optional[str]) -> dict:
        """
        Store data using the Cogniswitch service.
        This calls the CogniSwitch services to analyze & store data from a url.
        the URL is provided in input, assign that value to the url key.
        Assign document name & description only if provided in input.

        Args:
            url (Optional[str]): URL link.
            document_name (Optional[str]): Name of the document you are uploading.
            document_description (Optional[str]): Description of the document.

        Returns:
            dict: Response JSON from the Cogniswitch service.
        """
        headers = {'apiKey': self.apiKey, 'openAIToken': self.OAI_token, 'platformToken': self.cs_token}
        data: Dict[str, Any]
        if not document_name:
            document_name = ''
        if not document_description:
            document_description = ''
        if not url:
            return {'message': 'No input provided'}
        else:
            data = {'url': url}
            response = requests.post(self.knowledgesource_url, headers=headers, verify=False, data=data)
        if response.status_code == 200:
            return response.json()
        else:
            return {'message': 'Bad Request'}