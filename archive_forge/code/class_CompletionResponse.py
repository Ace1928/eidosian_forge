import time
from typing import Dict, List, Literal, Optional, Union
from pydantic import BaseModel, Field, model_validator
from vllm.utils import random_uuid
from vllm.sampling_params import SamplingParams
import torch
class CompletionResponse(BaseModel):
    id: str = Field(default_factory=lambda: f'cmpl-{random_uuid()}')
    object: str = 'text_completion'
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[CompletionResponseChoice]
    usage: UsageInfo