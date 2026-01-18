import random
from pathlib import Path
from typing import List
import numpy as np
import torch
from safetensors.torch import load_file
from torch.cuda.amp import GradScaler
from .utils import (
from .logging import get_logger
from .state import PartialState

    Loads the state of `obj` at `{path}/custom_checkpoint_{index}.pkl`
    