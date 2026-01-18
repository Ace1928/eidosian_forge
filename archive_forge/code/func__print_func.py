from typing import Callable, Optional
from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.pydantic_v1 import Field
from langchain_core.tools import BaseTool
def _print_func(text: str) -> None:
    print('\n')
    print(text)