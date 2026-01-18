from typing import Any, Dict, Optional
from langchain_core.pydantic_v1 import BaseModel, Field, root_validator
from langchain_core.language_models import BaseLanguageModel
from langchain_core.callbacks import (
from langchain_core.prompts import PromptTemplate
from langchain_community.utilities.spark_sql import SparkSQL
from langchain_core.tools import BaseTool
from langchain_community.tools.spark_sql.prompt import QUERY_CHECKER
class InfoSparkSQLTool(BaseSparkSQLTool, BaseTool):
    """Tool for getting metadata about a Spark SQL."""
    name: str = 'schema_sql_db'
    description: str = '\n    Input to this tool is a comma-separated list of tables, output is the schema and sample rows for those tables.\n    Be sure that the tables actually exist by calling list_tables_sql_db first!\n\n    Example Input: "table1, table2, table3"\n    '

    def _run(self, table_names: str, run_manager: Optional[CallbackManagerForToolRun]=None) -> str:
        """Get the schema for tables in a comma-separated list."""
        return self.db.get_table_info_no_throw(table_names.split(', '))