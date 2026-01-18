from operator import itemgetter
from typing import Any, Callable, List, Mapping, Optional, Union
from langchain_core.messages import BaseMessage
from langchain_core.output_parsers.openai_functions import JsonOutputFunctionsParser
from langchain_core.runnables import RouterRunnable, Runnable
from langchain_core.runnables.base import RunnableBindingBase
from typing_extensions import TypedDict
class OpenAIFunction(TypedDict):
    """A function description for ChatOpenAI"""
    name: str
    'The name of the function.'
    description: str
    'The description of the function.'
    parameters: dict
    'The parameters to the function.'