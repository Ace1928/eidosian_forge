import time
from typing import Dict, List, Literal, Optional, Union
from pydantic import BaseModel, Field
from vllm.utils import random_uuid
class LoggingRecord(BaseModel):
    time: int
    request: ChatCompletionRequest
    outputs: List[str]