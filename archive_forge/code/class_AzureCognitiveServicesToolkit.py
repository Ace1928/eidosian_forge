from __future__ import annotations
import sys
from typing import List
from langchain_core.tools import BaseTool
from langchain_community.agent_toolkits.base import BaseToolkit
from langchain_community.tools.azure_cognitive_services import (
class AzureCognitiveServicesToolkit(BaseToolkit):
    """Toolkit for Azure Cognitive Services."""

    def get_tools(self) -> List[BaseTool]:
        """Get the tools in the toolkit."""
        tools: List[BaseTool] = [AzureCogsFormRecognizerTool(), AzureCogsSpeech2TextTool(), AzureCogsText2SpeechTool(), AzureCogsTextAnalyticsHealthTool()]
        if sys.platform.startswith('linux') or sys.platform.startswith('win'):
            tools.append(AzureCogsImageAnalysisTool())
        return tools