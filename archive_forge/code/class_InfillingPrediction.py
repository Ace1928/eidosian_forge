import json
import os
import sys
import time
from pathlib import Path
from typing import List, Literal, Optional, Tuple, TypedDict
import torch
import torch.nn.functional as F
from fairscale.nn.model_parallel.initialize import (
from llama.model import ModelArgs, Transformer
from llama.tokenizer import Tokenizer
class InfillingPrediction(TypedDict, total=False):
    generation: str
    full_text: str
    tokens: List[str]
    logprobs: List[float]