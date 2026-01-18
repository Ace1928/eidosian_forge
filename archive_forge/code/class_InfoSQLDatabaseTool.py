from typing import Any, Dict, Optional, Sequence, Type, Union
from sqlalchemy.engine import Result
from langchain_core.pydantic_v1 import BaseModel, Field, root_validator
from langchain_core.language_models import BaseLanguageModel
from langchain_core.callbacks import (
from langchain_core.prompts import PromptTemplate
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_core.tools import BaseTool
from langchain_community.tools.sql_database.prompt import QUERY_CHECKER
class InfoSQLDatabaseTool(BaseSQLDatabaseTool, BaseTool):
    """Tool for getting metadata about a SQL database."""
    name: str = 'sql_db_schema'
    description: str = 'Get the schema and sample rows for the specified SQL tables.'
    args_schema: Type[BaseModel] = _InfoSQLDatabaseToolInput

    def _run(self, table_names: str, run_manager: Optional[CallbackManagerForToolRun]=None) -> str:
        """Get the schema for tables in a comma-separated list."""
        return self.db.get_table_info_no_throw([t.strip() for t in table_names.split(',')])