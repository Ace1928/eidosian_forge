import time
from typing import Dict, List, Literal, Optional, Union
from pydantic import BaseModel, Field, model_validator
from vllm.utils import random_uuid
from vllm.sampling_params import SamplingParams
import torch
class ModelList(BaseModel):
    object: str = 'list'
    data: List[ModelCard] = Field(default_factory=list)