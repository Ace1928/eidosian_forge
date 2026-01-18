import filelock
import glob
import fnmatch
import json
import os
from collections import defaultdict
from typing import Any, Iterator, List, Optional, Tuple
from huggingface_hub import snapshot_download, HfFileSystem
import numpy as np
from safetensors.torch import load_file, save_file, safe_open
import torch
from tqdm.auto import tqdm
from vllm.config import ModelConfig
from vllm.logger import init_logger
from vllm.model_executor.layers.quantization import (get_quantization_config,
def _shared_pointers(tensors):
    ptrs = defaultdict(list)
    for k, v in tensors.items():
        ptrs[v.data_ptr()].append(k)
    failing = []
    for _, names in ptrs.items():
        if len(names) > 1:
            failing.append(names)
    return failing