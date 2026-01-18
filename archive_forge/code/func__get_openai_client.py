from __future__ import annotations
import json
from json import JSONDecodeError
from time import sleep
from typing import (
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.callbacks import CallbackManager
from langchain_core.load import dumpd
from langchain_core.pydantic_v1 import BaseModel, Field, root_validator
from langchain_core.runnables import RunnableConfig, RunnableSerializable, ensure_config
from langchain_core.tools import BaseTool
from langchain_core.utils.function_calling import convert_to_openai_tool
def _get_openai_client() -> openai.OpenAI:
    try:
        import openai
        return openai.OpenAI()
    except ImportError as e:
        raise ImportError('Unable to import openai, please install with `pip install openai`.') from e
    except AttributeError as e:
        raise AttributeError('Please make sure you are using a v1.1-compatible version of openai. You can install with `pip install "openai>=1.1"`.') from e