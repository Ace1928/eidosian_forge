import builtins
import collections
import functools
import inspect
import math
import operator
import os
import random
import warnings
from typing import Any, Callable, Dict, List, Optional, Type, Union
import torch
from torch import nn
from torch.fx import Graph, GraphModule, Proxy, Tracer
from torch.fx._compatibility import compatibility
from torch.fx.proxy import ParameterProxy
from .. import PretrainedConfig, PreTrainedModel, logging
from ..models.auto import get_values
from ..models.auto.modeling_auto import (
from ..pytorch_utils import is_torch_greater_or_equal_than_2_0
from ..utils import (
def _generate_supported_model_class_names(model_name: Type[PretrainedConfig], supported_tasks: Optional[Union[str, List[str]]]=None) -> List[str]:
    task_mapping = {'default': MODEL_MAPPING_NAMES, 'pretraining': MODEL_FOR_PRETRAINING_MAPPING_NAMES, 'next-sentence-prediction': MODEL_FOR_NEXT_SENTENCE_PREDICTION_MAPPING_NAMES, 'masked-lm': MODEL_FOR_MASKED_LM_MAPPING_NAMES, 'causal-lm': MODEL_FOR_CAUSAL_LM_MAPPING_NAMES, 'seq2seq-lm': MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING_NAMES, 'speech-seq2seq': MODEL_FOR_SPEECH_SEQ_2_SEQ_MAPPING_NAMES, 'multiple-choice': MODEL_FOR_MULTIPLE_CHOICE_MAPPING_NAMES, 'document-question-answering': MODEL_FOR_DOCUMENT_QUESTION_ANSWERING_MAPPING_NAMES, 'question-answering': MODEL_FOR_QUESTION_ANSWERING_MAPPING_NAMES, 'sequence-classification': MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING_NAMES, 'token-classification': MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING_NAMES, 'masked-image-modeling': MODEL_FOR_MASKED_IMAGE_MODELING_MAPPING_NAMES, 'image-classification': MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING_NAMES, 'zero-shot-image-classification': MODEL_FOR_ZERO_SHOT_IMAGE_CLASSIFICATION_MAPPING_NAMES, 'ctc': MODEL_FOR_CTC_MAPPING_NAMES, 'audio-classification': MODEL_FOR_AUDIO_CLASSIFICATION_MAPPING_NAMES, 'semantic-segmentation': MODEL_FOR_SEMANTIC_SEGMENTATION_MAPPING_NAMES, 'backbone': MODEL_FOR_BACKBONE_MAPPING_NAMES}
    if supported_tasks is None:
        supported_tasks = task_mapping.keys()
    if isinstance(supported_tasks, str):
        supported_tasks = [supported_tasks]
    model_class_names = []
    for task in supported_tasks:
        class_name = task_mapping[task].get(model_name, None)
        if class_name:
            model_class_names.append(class_name)
    return model_class_names