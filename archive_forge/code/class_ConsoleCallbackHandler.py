import json
from typing import Any, Callable, List
from langchain_core.tracers.base import BaseTracer
from langchain_core.tracers.schemas import Run
from langchain_core.utils.input import get_bolded_text, get_colored_text
class ConsoleCallbackHandler(FunctionCallbackHandler):
    """Tracer that prints to the console."""
    name: str = 'console_callback_handler'

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(function=print, **kwargs)