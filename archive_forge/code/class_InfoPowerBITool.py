import logging
from time import perf_counter
from typing import Any, Dict, Optional, Tuple
from langchain_core.callbacks import (
from langchain_core.pydantic_v1 import Field, validator
from langchain_core.tools import BaseTool
from langchain_community.chat_models.openai import _import_tiktoken
from langchain_community.tools.powerbi.prompt import (
from langchain_community.utilities.powerbi import PowerBIDataset, json_to_md
class InfoPowerBITool(BaseTool):
    """Tool for getting metadata about a PowerBI Dataset."""
    name: str = 'schema_powerbi'
    description: str = '\n    Input to this tool is a comma-separated list of tables, output is the schema and sample rows for those tables.\n    Be sure that the tables actually exist by calling list_tables_powerbi first!\n\n    Example Input: "table1, table2, table3"\n    '
    powerbi: PowerBIDataset = Field(exclude=True)

    class Config:
        """Configuration for this pydantic object."""
        arbitrary_types_allowed = True

    def _run(self, tool_input: str, run_manager: Optional[CallbackManagerForToolRun]=None) -> str:
        """Get the schema for tables in a comma-separated list."""
        return self.powerbi.get_table_info(tool_input.split(', '))

    async def _arun(self, tool_input: str, run_manager: Optional[AsyncCallbackManagerForToolRun]=None) -> str:
        return await self.powerbi.aget_table_info(tool_input.split(', '))