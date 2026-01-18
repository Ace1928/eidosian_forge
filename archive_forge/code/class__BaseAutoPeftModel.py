from __future__ import annotations
import importlib
import os
from typing import Optional
from transformers import (
from .config import PeftConfig
from .mapping import MODEL_TYPE_TO_PEFT_MODEL_MAPPING
from .peft_model import (
from .utils.constants import TOKENIZER_CONFIG_NAME
from .utils.other import check_file_exists_on_hf_hub
class _BaseAutoPeftModel:
    _target_class = None
    _target_peft_class = None

    def __init__(self, *args, **kwargs):
        raise EnvironmentError(f'{self.__class__.__name__} is designed to be instantiated using the `{self.__class__.__name__}.from_pretrained(pretrained_model_name_or_path)` or `{self.__class__.__name__}.from_config(config)` methods.')

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, adapter_name: str='default', is_trainable: bool=False, config: Optional[PeftConfig]=None, **kwargs):
        """
        A wrapper around all the preprocessing steps a user needs to perform in order to load a PEFT model. The kwargs
        are passed along to `PeftConfig` that automatically takes care of filtering the kwargs of the Hub methods and
        the config object init.
        """
        peft_config = PeftConfig.from_pretrained(pretrained_model_name_or_path, **kwargs)
        base_model_path = peft_config.base_model_name_or_path
        task_type = getattr(peft_config, 'task_type', None)
        if cls._target_class is not None:
            target_class = cls._target_class
        elif cls._target_class is None and task_type is not None:
            raise ValueError("Cannot use `AutoPeftModel` with a task type, please use a specific class for your task type. (e.g. `AutoPeftModelForCausalLM` for `task_type='CAUSAL_LM'`)")
        if task_type is not None:
            expected_target_class = MODEL_TYPE_TO_PEFT_MODEL_MAPPING[task_type]
            if cls._target_peft_class.__name__ != expected_target_class.__name__:
                raise ValueError(f'Expected target PEFT class: {expected_target_class.__name__}, but you have asked for: {cls._target_peft_class.__name__} make sure that you are loading the correct model for your task type.')
        elif task_type is None and getattr(peft_config, 'auto_mapping', None) is not None:
            auto_mapping = getattr(peft_config, 'auto_mapping', None)
            base_model_class = auto_mapping['base_model_class']
            parent_library_name = auto_mapping['parent_library']
            parent_library = importlib.import_module(parent_library_name)
            target_class = getattr(parent_library, base_model_class)
        else:
            raise ValueError('Cannot infer the auto class from the config, please make sure that you are loading the correct model for your task type.')
        base_model = target_class.from_pretrained(base_model_path, **kwargs)
        tokenizer_exists = False
        if os.path.exists(os.path.join(pretrained_model_name_or_path, TOKENIZER_CONFIG_NAME)):
            tokenizer_exists = True
        else:
            token = kwargs.get('token', None)
            if token is None:
                token = kwargs.get('use_auth_token', None)
            tokenizer_exists = check_file_exists_on_hf_hub(repo_id=pretrained_model_name_or_path, filename=TOKENIZER_CONFIG_NAME, revision=kwargs.get('revision', None), repo_type=kwargs.get('repo_type', None), token=token)
        if tokenizer_exists:
            tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path, trust_remote_code=kwargs.get('trust_remote_code', False))
            base_model.resize_token_embeddings(len(tokenizer))
        return cls._target_peft_class.from_pretrained(base_model, pretrained_model_name_or_path, adapter_name=adapter_name, is_trainable=is_trainable, config=config, **kwargs)