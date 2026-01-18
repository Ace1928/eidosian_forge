import os
from typing import Any, Dict, List
from uuid import UUID
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import BaseMessage
from langchain_core.outputs import LLMResult
Log the conversation to the context API.