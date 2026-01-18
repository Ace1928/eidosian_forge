import time
from typing import Dict, List, Literal, Optional, Union
from pydantic import BaseModel, Field, model_validator
from vllm.utils import random_uuid
from vllm.sampling_params import SamplingParams
import torch
class CompletionResponseStreamChoice(BaseModel):
    index: int
    text: str
    logprobs: Optional[LogProbs] = None
    finish_reason: Optional[Literal['stop', 'length']] = None