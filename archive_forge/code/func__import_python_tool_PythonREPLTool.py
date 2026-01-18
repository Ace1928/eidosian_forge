import warnings
from typing import Any
from langchain_core._api import LangChainDeprecationWarning
from langchain_core.tools import BaseTool, StructuredTool, Tool, tool
from langchain.utils.interactive_env import is_interactive_env
def _import_python_tool_PythonREPLTool() -> Any:
    raise ImportError("This tool has been moved to langchain experiment. This tool has access to a python REPL. For best practices make sure to sandbox this tool. Read https://github.com/langchain-ai/langchain/blob/master/SECURITY.md To keep using this code as is, install langchain experimental and update relevant imports replacing 'langchain' with 'langchain_experimental'")