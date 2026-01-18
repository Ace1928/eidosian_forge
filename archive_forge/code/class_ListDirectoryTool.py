import os
from typing import Optional, Type
from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.tools import BaseTool
from langchain_community.tools.file_management.utils import (
class ListDirectoryTool(BaseFileToolMixin, BaseTool):
    """Tool that lists files and directories in a specified folder."""
    name: str = 'list_directory'
    args_schema: Type[BaseModel] = DirectoryListingInput
    description: str = 'List files and directories in a specified folder'

    def _run(self, dir_path: str='.', run_manager: Optional[CallbackManagerForToolRun]=None) -> str:
        try:
            dir_path_ = self.get_relative_path(dir_path)
        except FileValidationError:
            return INVALID_PATH_TEMPLATE.format(arg_name='dir_path', value=dir_path)
        try:
            entries = os.listdir(dir_path_)
            if entries:
                return '\n'.join(entries)
            else:
                return f'No files found in directory {dir_path}'
        except Exception as e:
            return 'Error: ' + str(e)