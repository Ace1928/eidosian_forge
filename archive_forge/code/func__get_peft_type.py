import inspect
import json
import os
from dataclasses import asdict, dataclass, field
from typing import Dict, Optional, Union
from huggingface_hub import hf_hub_download
from transformers.utils import PushToHubMixin
from .utils import CONFIG_NAME, PeftType, TaskType
@classmethod
def _get_peft_type(cls, model_id: str, **hf_hub_download_kwargs):
    subfolder = hf_hub_download_kwargs.get('subfolder', None)
    path = os.path.join(model_id, subfolder) if subfolder is not None else model_id
    if os.path.isfile(os.path.join(path, CONFIG_NAME)):
        config_file = os.path.join(path, CONFIG_NAME)
    else:
        try:
            config_file = hf_hub_download(model_id, CONFIG_NAME, **hf_hub_download_kwargs)
        except Exception:
            raise ValueError(f"Can't find '{CONFIG_NAME}' at '{model_id}'")
    loaded_attributes = cls.from_json_file(config_file)
    return loaded_attributes['peft_type']