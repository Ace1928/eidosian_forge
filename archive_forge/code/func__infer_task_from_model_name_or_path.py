import importlib
import inspect
import itertools
import os
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Set, Tuple, Type, Union
import huggingface_hub
from packaging import version
from requests.exceptions import ConnectionError as RequestsConnectionError
from transformers import AutoConfig, PretrainedConfig, is_tf_available, is_torch_available
from transformers.utils import SAFE_WEIGHTS_NAME, TF2_WEIGHTS_NAME, WEIGHTS_NAME, logging
from ..utils import CONFIG_NAME
from ..utils.import_utils import is_onnx_available
@classmethod
def _infer_task_from_model_name_or_path(cls, model_name_or_path: str, subfolder: str='', revision: Optional[str]=None) -> str:
    inferred_task_name = None
    is_local = os.path.isdir(os.path.join(model_name_or_path, subfolder))
    if is_local:
        raise RuntimeError(f'Cannot infer the task from a local directory yet, please specify the task manually ({', '.join(TasksManager.get_all_tasks())}).')
    else:
        if subfolder != '':
            raise RuntimeError('Cannot infer the task from a model repo with a subfolder yet, please specify the task manually.')
        model_info = huggingface_hub.model_info(model_name_or_path, revision=revision)
        library_name = TasksManager.infer_library_from_model(model_name_or_path, subfolder, revision)
        if library_name == 'diffusers':
            class_name = model_info.config['diffusers']['class_name']
            inferred_task_name = 'stable-diffusion-xl' if 'StableDiffusionXL' in class_name else 'stable-diffusion'
        elif library_name == 'timm':
            inferred_task_name = 'image-classification'
        else:
            pipeline_tag = getattr(model_info, 'pipeline_tag', None)
            if pipeline_tag is not None and pipeline_tag not in ['conversational', 'object-detection']:
                inferred_task_name = TasksManager.map_from_synonym(model_info.pipeline_tag)
            else:
                transformers_info = model_info.transformersInfo
                if transformers_info is not None and transformers_info.get('pipeline_tag') is not None:
                    inferred_task_name = TasksManager.map_from_synonym(transformers_info['pipeline_tag'])
                else:
                    class_name_prefix = ''
                    if is_torch_available():
                        tasks_to_automodels = TasksManager._LIBRARY_TO_TASKS_TO_MODEL_LOADER_MAP[library_name]
                    else:
                        tasks_to_automodels = TasksManager._LIBRARY_TO_TF_TASKS_TO_MODEL_LOADER_MAP[library_name]
                        class_name_prefix = 'TF'
                    auto_model_class_name = transformers_info['auto_model']
                    if not auto_model_class_name.startswith('TF'):
                        auto_model_class_name = f'{class_name_prefix}{auto_model_class_name}'
                    for task_name, class_name_for_task in tasks_to_automodels.items():
                        if class_name_for_task == auto_model_class_name:
                            inferred_task_name = task_name
                            break
    if inferred_task_name is None:
        raise KeyError(f'Could not find the proper task name for {auto_model_class_name}.')
    return inferred_task_name