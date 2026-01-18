from __future__ import annotations
from typing import Any, Dict, Optional
import requests
from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.tools import BaseTool
class CogniswitchKnowledgeSourceFile(BaseTool):
    """Tool that uses the Cogniswitch services to store data from file.

    name: str = "cogniswitch_knowledge_source_file"
    description: str = (
        "This calls the CogniSwitch services to analyze & store data from a file.
        If the input looks like a file path, assign that string value to file key.
        Assign document name & description only if provided in input."
    )
    """
    name: str = 'cogniswitch_knowledge_source_file'
    description: str = '\n        This calls the CogniSwitch services to analyze & store data from a file. \n        If the input looks like a file path, assign that string value to file key. \n        Assign document name & description only if provided in input.\n        '
    cs_token: str
    OAI_token: str
    apiKey: str
    knowledgesource_file = 'https://api.cogniswitch.ai:8243/cs-api/0.0.1/cs/knowledgeSource/file'

    def _run(self, file: Optional[str]=None, document_name: Optional[str]=None, document_description: Optional[str]=None, run_manager: Optional[CallbackManagerForToolRun]=None) -> Dict[str, Any]:
        """
        Execute the tool to store the data given from a file.
        This calls the CogniSwitch services to analyze & store data from a file.
        If the input looks like a file path, assign that string value to file key.
        Assign document name & description only if provided in input.

        Args:
            file Optional[str]: The file path of your knowledge
            document_name Optional[str]: Name of your knowledge document
            document_description Optional[str]: Description of your knowledge document
            run_manager (Optional[CallbackManagerForChainRun]):
            Manager for chain run callbacks.

        Returns:
            Dict[str, Any]: Output dictionary containing
            the 'response' from the service.
        """
        if not file:
            return {'message': 'No input provided'}
        else:
            response = self.store_data(file=file, document_name=document_name, document_description=document_description)
            return response

    def store_data(self, file: Optional[str], document_name: Optional[str], document_description: Optional[str]) -> dict:
        """
        Store data using the Cogniswitch service.
        This calls the CogniSwitch services to analyze & store data from a file.
        If the input looks like a file path, assign that string value to file key.
        Assign document name & description only if provided in input.

        Args:
            file (Optional[str]): file path of your file.
            the current files supported by the files are
            .txt, .pdf, .docx, .doc, .html
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
        if file is not None:
            files = {'file': open(file, 'rb')}
        data = {'documentName': document_name, 'documentDescription': document_description}
        response = requests.post(self.knowledgesource_file, headers=headers, verify=False, data=data, files=files)
        if response.status_code == 200:
            return response.json()
        else:
            return {'message': 'Bad Request'}