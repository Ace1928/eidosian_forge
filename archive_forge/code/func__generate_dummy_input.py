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
def _generate_dummy_input(self, model: PreTrainedModel, input_name: str, shape: List[int], input_names: List[str]) -> Dict[str, torch.Tensor]:
    """Generates dummy input for model inference recording."""
    model_class_name = getattr(model, 'class_for_deserialization', model.__class__).__name__
    device = model.device
    inputs_dict = {}
    kv_cache_length = 5
    if input_name in ['labels', 'start_positions', 'end_positions']:
        batch_size = shape[0]
        if model_class_name in [*get_values(MODEL_FOR_NEXT_SENTENCE_PREDICTION_MAPPING_NAMES), *get_values(MODEL_FOR_MULTIPLE_CHOICE_MAPPING_NAMES), *get_values(MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING_NAMES), *get_values(MODEL_FOR_BACKBONE_MAPPING_NAMES), *get_values(MODEL_FOR_AUDIO_CLASSIFICATION_MAPPING_NAMES)]:
            inputs_dict['labels'] = torch.zeros(batch_size, dtype=torch.long, device=device)
        elif model_class_name in [*get_values(MODEL_FOR_QUESTION_ANSWERING_MAPPING_NAMES), *get_values(MODEL_FOR_DOCUMENT_QUESTION_ANSWERING_MAPPING_NAMES), 'XLNetForQuestionAnswering']:
            inputs_dict['start_positions'] = torch.zeros(batch_size, dtype=torch.long, device=device)
            inputs_dict['end_positions'] = torch.zeros(batch_size, dtype=torch.long, device=device)
        elif model_class_name in get_values(MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING_NAMES):
            if not hasattr(model.config, 'problem_type') or model.config.problem_type is None:
                raise ValueError('Could not retrieve the problem type for the sequence classification task, please set model.config.problem_type to one of the following values: "regression", "single_label_classification", or "multi_label_classification".')
            if model.config.problem_type == 'regression':
                labels_shape = (batch_size, model.config.num_labels)
                labels_dtype = torch.float32
            elif model.config.problem_type == 'single_label_classification':
                labels_shape = (batch_size,)
                labels_dtype = torch.long
            elif model.config.problem_type == 'multi_label_classification':
                labels_shape = (batch_size, model.config.num_labels)
                labels_dtype = torch.float32
            else:
                raise ValueError(f'Expected model.config.problem_type to be either: "regression", "single_label_classification", or "multi_label_classification", but "{model.config.problem_type}" was provided.')
            inputs_dict['labels'] = torch.zeros(*labels_shape, dtype=labels_dtype, device=device)
        elif model_class_name in [*get_values(MODEL_FOR_PRETRAINING_MAPPING_NAMES), *get_values(MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING_NAMES), *get_values(MODEL_FOR_CAUSAL_LM_MAPPING_NAMES), *get_values(MODEL_FOR_MASKED_LM_MAPPING_NAMES), *get_values(MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING_NAMES), *get_values(MODEL_FOR_SEMANTIC_SEGMENTATION_MAPPING_NAMES), 'GPT2DoubleHeadsModel', 'PeftModelForCausalLM', 'PeftModelForSeq2SeqLM']:
            inputs_dict['labels'] = torch.zeros(shape, dtype=torch.long, device=device)
        elif model_class_name in [*get_values(MODEL_FOR_CTC_MAPPING_NAMES)]:
            inputs_dict['labels'] = torch.zeros(shape, dtype=torch.float32, device=device)
        else:
            raise NotImplementedError(f'Generating the dummy input named {input_name} for {model_class_name} is not supported yet.')
    elif 'pixel_values' in input_name:
        batch_size = shape[0]
        image_size = getattr(model.config, 'image_size', None)
        if image_size is None:
            if hasattr(model.config, 'vision_config'):
                image_size = model.config.vision_config.image_size
            elif hasattr(model.config, 'encoder'):
                image_size = model.config.encoder.image_size
            else:
                image_size = (_generate_random_int(), _generate_random_int())
        num_channels = getattr(model.config, 'num_channels', 3)
        if not isinstance(image_size, collections.abc.Iterable):
            image_size = (image_size, image_size)
        height, width = image_size
        inputs_dict[input_name] = torch.zeros(batch_size, num_channels, height, width, dtype=torch.float32, device=device)
    elif 'bbox' in input_name:
        inputs_dict[input_name] = torch.zeros(*shape, 4, dtype=torch.float, device=device)
    elif 'input_features' in input_name:
        inputs_dict[input_name] = torch.zeros(*shape, model.config.input_feat_per_channel, dtype=torch.float, device=device)
    elif 'visual_feats' in input_name:
        inputs_dict[input_name] = torch.zeros(shape + [model.config.visual_feat_dim], dtype=torch.float, device=device)
    elif 'visual_pos' in input_name:
        inputs_dict[input_name] = torch.zeros(shape + [model.config.visual_pos_dim], dtype=torch.float, device=device)
    elif 'inputs' in input_name:
        inputs_dict[input_name] = torch.zeros(*shape, dtype=torch.float, device=device)
    elif 'input_values' in input_name:
        batch_size, _ = shape
        seq_length = _generate_random_int(low=10000, high=20000)
        inputs_dict[input_name] = torch.zeros(batch_size, seq_length, dtype=torch.float, device=device)
    elif 'mask' in input_name:
        if 'past_key_values' in input_names:
            mask_shape = [shape[0], shape[1] + kv_cache_length]
        else:
            mask_shape = shape
        inputs_dict[input_name] = torch.zeros(mask_shape, dtype=torch.long, device=device)
    elif 'ids' in input_name:
        inputs_dict[input_name] = torch.zeros(shape, dtype=torch.long, device=device)
    elif 'past_key_values' in input_name:
        if model.config.model_type not in _FX_SUPPORTED_MODELS_WITH_KV_CACHE:
            raise NotImplementedError(f'Symbolic trace with past_key_values input is not supported yet for the model {model.config.model_type}. Please open an issue or a PR in Transformers repository if you would like to see the support added.')
        num_heads = model.config.num_attention_heads
        head_dim = model.config.hidden_size // model.config.num_attention_heads
        cache_shape = (shape[0], num_heads, kv_cache_length, head_dim)
        pkv = tuple(((torch.rand(cache_shape, dtype=torch.float, device=device), torch.rand(cache_shape, dtype=torch.float, device=device)) for i in range(model.config.num_hidden_layers)))
        inputs_dict[input_name] = pkv
    else:
        shape_with_hidden_size = shape + [model.config.hidden_size]
        inputs_dict[input_name] = torch.zeros(shape_with_hidden_size, dtype=torch.float, device=device)
    return inputs_dict