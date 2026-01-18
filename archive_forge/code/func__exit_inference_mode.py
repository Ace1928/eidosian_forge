from typing import Any, Optional
import torch
from torch.utils._contextlib import (
def _exit_inference_mode(mode):
    mode.__exit__(None, None, None)