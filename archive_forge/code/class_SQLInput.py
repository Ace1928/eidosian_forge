from typing import Any, Dict, List, Optional, TypedDict, Union
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_core.language_models import BaseLanguageModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import BasePromptTemplate
from langchain_core.runnables import Runnable, RunnablePassthrough
from langchain.chains.sql_database.prompt import PROMPT, SQL_PROMPTS
class SQLInput(TypedDict):
    """Input for a SQL Chain."""
    question: str