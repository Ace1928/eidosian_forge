from operator import itemgetter
from typing import Any, Callable, List, Mapping, Optional, Union
from langchain_core.messages import BaseMessage
from langchain_core.output_parsers.openai_functions import JsonOutputFunctionsParser
from langchain_core.runnables import RouterRunnable, Runnable
from langchain_core.runnables.base import RunnableBindingBase
from typing_extensions import TypedDict
class OpenAIFunctionsRouter(RunnableBindingBase[BaseMessage, Any]):
    """A runnable that routes to the selected function."""
    functions: Optional[List[OpenAIFunction]]

    def __init__(self, runnables: Mapping[str, Union[Runnable[dict, Any], Callable[[dict], Any]]], functions: Optional[List[OpenAIFunction]]=None):
        if functions is not None:
            assert len(functions) == len(runnables)
            assert all((func['name'] in runnables for func in functions))
        router = JsonOutputFunctionsParser(args_only=False) | {'key': itemgetter('name'), 'input': itemgetter('arguments')} | RouterRunnable(runnables)
        super().__init__(bound=router, kwargs={}, functions=functions)