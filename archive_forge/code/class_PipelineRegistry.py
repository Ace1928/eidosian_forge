import collections
import csv
import importlib
import json
import os
import pickle
import sys
import traceback
import types
import warnings
from abc import ABC, abstractmethod
from collections import UserDict
from contextlib import contextmanager
from os.path import abspath, exists
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union
from ..dynamic_module_utils import custom_object_save
from ..feature_extraction_utils import PreTrainedFeatureExtractor
from ..image_processing_utils import BaseImageProcessor
from ..modelcard import ModelCard
from ..models.auto.configuration_auto import AutoConfig
from ..tokenization_utils import PreTrainedTokenizer
from ..utils import (
class PipelineRegistry:

    def __init__(self, supported_tasks: Dict[str, Any], task_aliases: Dict[str, str]) -> None:
        self.supported_tasks = supported_tasks
        self.task_aliases = task_aliases

    def get_supported_tasks(self) -> List[str]:
        supported_task = list(self.supported_tasks.keys()) + list(self.task_aliases.keys())
        supported_task.sort()
        return supported_task

    def check_task(self, task: str) -> Tuple[str, Dict, Any]:
        if task in self.task_aliases:
            task = self.task_aliases[task]
        if task in self.supported_tasks:
            targeted_task = self.supported_tasks[task]
            return (task, targeted_task, None)
        if task.startswith('translation'):
            tokens = task.split('_')
            if len(tokens) == 4 and tokens[0] == 'translation' and (tokens[2] == 'to'):
                targeted_task = self.supported_tasks['translation']
                task = 'translation'
                return (task, targeted_task, (tokens[1], tokens[3]))
            raise KeyError(f"Invalid translation task {task}, use 'translation_XX_to_YY' format")
        raise KeyError(f'Unknown task {task}, available tasks are {self.get_supported_tasks() + ['translation_XX_to_YY']}')

    def register_pipeline(self, task: str, pipeline_class: type, pt_model: Optional[Union[type, Tuple[type]]]=None, tf_model: Optional[Union[type, Tuple[type]]]=None, default: Optional[Dict]=None, type: Optional[str]=None) -> None:
        if task in self.supported_tasks:
            logger.warning(f'{task} is already registered. Overwriting pipeline for task {task}...')
        if pt_model is None:
            pt_model = ()
        elif not isinstance(pt_model, tuple):
            pt_model = (pt_model,)
        if tf_model is None:
            tf_model = ()
        elif not isinstance(tf_model, tuple):
            tf_model = (tf_model,)
        task_impl = {'impl': pipeline_class, 'pt': pt_model, 'tf': tf_model}
        if default is not None:
            if 'model' not in default and ('pt' in default or 'tf' in default):
                default = {'model': default}
            task_impl['default'] = default
        if type is not None:
            task_impl['type'] = type
        self.supported_tasks[task] = task_impl
        pipeline_class._registered_impl = {task: task_impl}

    def to_dict(self):
        return self.supported_tasks