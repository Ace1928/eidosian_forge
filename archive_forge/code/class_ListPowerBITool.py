import logging
from time import perf_counter
from typing import Any, Dict, Optional, Tuple
from langchain_core.callbacks import (
from langchain_core.pydantic_v1 import Field, validator
from langchain_core.tools import BaseTool
from langchain_community.chat_models.openai import _import_tiktoken
from langchain_community.tools.powerbi.prompt import (
from langchain_community.utilities.powerbi import PowerBIDataset, json_to_md
class ListPowerBITool(BaseTool):
    """Tool for getting tables names."""
    name: str = 'list_tables_powerbi'
    description: str = 'Input is an empty string, output is a comma separated list of tables in the database.'
    powerbi: PowerBIDataset = Field(exclude=True)

    class Config:
        """Configuration for this pydantic object."""
        arbitrary_types_allowed = True

    def _run(self, tool_input: Optional[str]=None, run_manager: Optional[CallbackManagerForToolRun]=None) -> str:
        """Get the names of the tables."""
        return ', '.join(self.powerbi.get_table_names())

    async def _arun(self, tool_input: Optional[str]=None, run_manager: Optional[AsyncCallbackManagerForToolRun]=None) -> str:
        """Get the names of the tables."""
        return ', '.join(self.powerbi.get_table_names())