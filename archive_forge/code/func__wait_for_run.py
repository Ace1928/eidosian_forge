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
def _wait_for_run(self, run_id: str, thread_id: str) -> Any:
    in_progress = True
    while in_progress:
        run = self.client.beta.threads.runs.retrieve(run_id, thread_id=thread_id)
        in_progress = run.status in ('in_progress', 'queued')
        if in_progress:
            sleep(self.check_every_ms / 1000)
    return run