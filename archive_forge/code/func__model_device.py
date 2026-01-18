import importlib.util
import json
import os
import time
from dataclasses import dataclass
from typing import Dict
import requests
from huggingface_hub import HfFolder, hf_hub_download, list_spaces
from ..models.auto import AutoTokenizer
from ..utils import is_offline_mode, is_openai_available, is_torch_available, logging
from .base import TASK_MAPPING, TOOL_CONFIG_FILE, Tool, load_tool, supports_remote
from .prompts import CHAT_MESSAGE_PROMPT, download_prompt
from .python_interpreter import evaluate
@property
def _model_device(self):
    if hasattr(self.model, 'hf_device_map'):
        return list(self.model.hf_device_map.values())[0]
    for param in self.model.parameters():
        return param.device