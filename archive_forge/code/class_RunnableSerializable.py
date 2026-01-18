from __future__ import annotations
import asyncio
import collections
import inspect
import threading
from abc import ABC, abstractmethod
from concurrent.futures import FIRST_COMPLETED, wait
from contextvars import copy_context
from functools import wraps
from itertools import groupby, tee
from operator import itemgetter
from typing import (
from typing_extensions import Literal, get_args
from langchain_core._api import beta_decorator
from langchain_core.load.dump import dumpd
from langchain_core.load.serializable import (
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.runnables.config import (
from langchain_core.runnables.graph import Graph
from langchain_core.runnables.schema import EventData, StreamEvent
from langchain_core.runnables.utils import (
from langchain_core.utils.aiter import atee, py_anext
from langchain_core.utils.iter import safetee
class RunnableSerializable(Serializable, Runnable[Input, Output]):
    """Runnable that can be serialized to JSON."""
    name: Optional[str] = None
    'The name of the runnable. Used for debugging and tracing.'

    def to_json(self) -> Union[SerializedConstructor, SerializedNotImplemented]:
        """Serialize the runnable to JSON."""
        dumped = super().to_json()
        try:
            dumped['name'] = self.get_name()
            dumped['graph'] = self.get_graph().to_json()
        except Exception:
            pass
        return dumped

    def configurable_fields(self, **kwargs: AnyConfigurableField) -> RunnableSerializable[Input, Output]:
        """Configure particular runnable fields at runtime.

        .. code-block:: python

            from langchain_core.runnables import ConfigurableField
            from langchain_openai import ChatOpenAI

            model = ChatOpenAI(max_tokens=20).configurable_fields(
                max_tokens=ConfigurableField(
                    id="output_token_number",
                    name="Max tokens in the output",
                    description="The maximum number of tokens in the output",
                )
            )

            # max_tokens = 20
            print(
                "max_tokens_20: ",
                model.invoke("tell me something about chess").content
            )

            # max_tokens = 200
            print("max_tokens_200: ", model.with_config(
                configurable={"output_token_number": 200}
                ).invoke("tell me something about chess").content
            )
        """
        from langchain_core.runnables.configurable import RunnableConfigurableFields
        for key in kwargs:
            if key not in self.__fields__:
                raise ValueError(f'Configuration key {key} not found in {self}: available keys are {self.__fields__.keys()}')
        return RunnableConfigurableFields(default=self, fields=kwargs)

    def configurable_alternatives(self, which: ConfigurableField, *, default_key: str='default', prefix_keys: bool=False, **kwargs: Union[Runnable[Input, Output], Callable[[], Runnable[Input, Output]]]) -> RunnableSerializable[Input, Output]:
        """Configure alternatives for runnables that can be set at runtime.

        .. code-block:: python

            from langchain_anthropic import ChatAnthropic
            from langchain_core.runnables.utils import ConfigurableField
            from langchain_openai import ChatOpenAI

            model = ChatAnthropic(
                model_name="claude-3-sonnet-20240229"
            ).configurable_alternatives(
                ConfigurableField(id="llm"),
                default_key="anthropic",
                openai=ChatOpenAI()
            )

            # uses the default model ChatAnthropic
            print(model.invoke("which organization created you?").content)

            # uses ChatOpenaAI
            print(
                model.with_config(
                    configurable={"llm": "openai"}
                ).invoke("which organization created you?").content
            )
        """
        from langchain_core.runnables.configurable import RunnableConfigurableAlternatives
        return RunnableConfigurableAlternatives(which=which, default=self, alternatives=kwargs, default_key=default_key, prefix_keys=prefix_keys)