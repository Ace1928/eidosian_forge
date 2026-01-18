from langchain_community.agent_toolkits.jira.toolkit import JiraToolkit
from langchain_community.utilities.jira import JiraAPIWrapper
from typing import Optional
from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.pydantic_v1 import Field
from langchain_core.tools import BaseTool
from langchain_community.utilities.jira import JiraAPIWrapper
class JiraAction(BaseTool):
    """Tool that queries the Atlassian Jira API."""
    api_wrapper: JiraAPIWrapper = Field(default_factory=JiraAPIWrapper)
    mode: str
    name: str = ''
    description: str = ''

    def _run(self, instructions: str, run_manager: Optional[CallbackManagerForToolRun]=None) -> str:
        """Use the Atlassian Jira API to run an operation."""
        return self.api_wrapper.run(self.mode, instructions)