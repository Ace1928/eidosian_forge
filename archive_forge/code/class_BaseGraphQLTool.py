import json
from typing import Optional
from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.tools import BaseTool
from langchain_community.utilities.graphql import GraphQLAPIWrapper
class BaseGraphQLTool(BaseTool):
    """Base tool for querying a GraphQL API."""
    graphql_wrapper: GraphQLAPIWrapper
    name: str = 'query_graphql'
    description: str = "    Input to this tool is a detailed and correct GraphQL query, output is a result from the API.\n    If the query is not correct, an error message will be returned.\n    If an error is returned with 'Bad request' in it, rewrite the query and try again.\n    If an error is returned with 'Unauthorized' in it, do not try again, but tell the user to change their authentication.\n\n    Example Input: query {{ allUsers {{ id, name, email }} }}    "

    class Config:
        """Configuration for this pydantic object."""
        arbitrary_types_allowed = True

    def _run(self, tool_input: str, run_manager: Optional[CallbackManagerForToolRun]=None) -> str:
        result = self.graphql_wrapper.run(tool_input)
        return json.dumps(result, indent=2)