import copy
import inspect
import json
import os
import warnings
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union
from .dynamic_module_utils import custom_object_save
from .tokenization_utils_base import PreTrainedTokenizerBase
from .utils import (
@classmethod
def _get_arguments_from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
    args = []
    for attribute_name in cls.attributes:
        class_name = getattr(cls, f'{attribute_name}_class')
        if isinstance(class_name, tuple):
            classes = tuple((getattr(transformers_module, n) if n is not None else None for n in class_name))
            use_fast = kwargs.get('use_fast', True)
            if use_fast and classes[1] is not None:
                attribute_class = classes[1]
            else:
                attribute_class = classes[0]
        else:
            attribute_class = getattr(transformers_module, class_name)
        args.append(attribute_class.from_pretrained(pretrained_model_name_or_path, **kwargs))
    return args