import inspect
import json
import os
from dataclasses import asdict, dataclass, field
from typing import Dict, Optional, Union
from huggingface_hub import hf_hub_download
from transformers.utils import PushToHubMixin
from .utils import CONFIG_NAME, PeftType, TaskType
@dataclass
class PeftConfig(PeftConfigMixin):
    """
    This is the base configuration class to store the configuration of a [`PeftModel`].

    Args:
        peft_type (Union[[`~peft.utils.config.PeftType`], `str`]): The type of Peft method to use.
        task_type (Union[[`~peft.utils.config.TaskType`], `str`]): The type of task to perform.
        inference_mode (`bool`, defaults to `False`): Whether to use the Peft model in inference mode.
    """
    base_model_name_or_path: Optional[str] = field(default=None, metadata={'help': 'The name of the base model to use.'})
    revision: Optional[str] = field(default=None, metadata={'help': 'The specific model version to use.'})
    peft_type: Optional[Union[str, PeftType]] = field(default=None, metadata={'help': 'Peft type'})
    task_type: Optional[Union[str, TaskType]] = field(default=None, metadata={'help': 'Task type'})
    inference_mode: bool = field(default=False, metadata={'help': 'Whether to use inference mode'})