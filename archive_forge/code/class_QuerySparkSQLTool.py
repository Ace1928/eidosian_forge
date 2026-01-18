from typing import Any, Dict, Optional
from langchain_core.pydantic_v1 import BaseModel, Field, root_validator
from langchain_core.language_models import BaseLanguageModel
from langchain_core.callbacks import (
from langchain_core.prompts import PromptTemplate
from langchain_community.utilities.spark_sql import SparkSQL
from langchain_core.tools import BaseTool
from langchain_community.tools.spark_sql.prompt import QUERY_CHECKER
class QuerySparkSQLTool(BaseSparkSQLTool, BaseTool):
    """Tool for querying a Spark SQL."""
    name: str = 'query_sql_db'
    description: str = '\n    Input to this tool is a detailed and correct SQL query, output is a result from the Spark SQL.\n    If the query is not correct, an error message will be returned.\n    If an error is returned, rewrite the query, check the query, and try again.\n    '

    def _run(self, query: str, run_manager: Optional[CallbackManagerForToolRun]=None) -> str:
        """Execute the query, return the results or an error message."""
        return self.db.run_no_throw(query)