from typing import Any, Dict, Optional, Sequence, Type, Union
from sqlalchemy.engine import Result
from langchain_core.pydantic_v1 import BaseModel, Field, root_validator
from langchain_core.language_models import BaseLanguageModel
from langchain_core.callbacks import (
from langchain_core.prompts import PromptTemplate
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_core.tools import BaseTool
from langchain_community.tools.sql_database.prompt import QUERY_CHECKER
class QuerySQLDataBaseTool(BaseSQLDatabaseTool, BaseTool):
    """Tool for querying a SQL database."""
    name: str = 'sql_db_query'
    description: str = '\n    Execute a SQL query against the database and get back the result..\n    If the query is not correct, an error message will be returned.\n    If an error is returned, rewrite the query, check the query, and try again.\n    '
    args_schema: Type[BaseModel] = _QuerySQLDataBaseToolInput

    def _run(self, query: str, run_manager: Optional[CallbackManagerForToolRun]=None) -> Union[str, Sequence[Dict[str, Any]], Result]:
        """Execute the query, return the results or an error message."""
        return self.db.run_no_throw(query)