from typing import Any, Dict, Optional, Sequence, Type, Union
from sqlalchemy.engine import Result
from langchain_core.pydantic_v1 import BaseModel, Field, root_validator
from langchain_core.language_models import BaseLanguageModel
from langchain_core.callbacks import (
from langchain_core.prompts import PromptTemplate
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_core.tools import BaseTool
from langchain_community.tools.sql_database.prompt import QUERY_CHECKER
class _QuerySQLCheckerToolInput(BaseModel):
    query: str = Field(..., description='A detailed and SQL query to be checked.')