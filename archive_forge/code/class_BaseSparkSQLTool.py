from typing import Any, Dict, Optional
from langchain_core.pydantic_v1 import BaseModel, Field, root_validator
from langchain_core.language_models import BaseLanguageModel
from langchain_core.callbacks import (
from langchain_core.prompts import PromptTemplate
from langchain_community.utilities.spark_sql import SparkSQL
from langchain_core.tools import BaseTool
from langchain_community.tools.spark_sql.prompt import QUERY_CHECKER
class BaseSparkSQLTool(BaseModel):
    """Base tool for interacting with Spark SQL."""
    db: SparkSQL = Field(exclude=True)

    class Config(BaseTool.Config):
        pass