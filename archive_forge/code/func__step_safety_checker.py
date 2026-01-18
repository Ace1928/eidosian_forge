import inspect
import math
import os
import time
import typing
import warnings
from contextlib import nullcontext
from typing import Callable, List, Optional, Union
import datasets
import numpy as np
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration, gather_object, is_deepspeed_available
from datasets import Dataset
from huggingface_hub import whoami
from packaging import version
from torch.optim import Adam
from transformers import (
from ..core import (
from ..import_utils import is_npu_available, is_torch_greater_2_0, is_xpu_available
from ..models import SUPPORTED_ARCHITECTURES, PreTrainedModelWrapper, create_reference_model
from . import AdaptiveKLController, BaseTrainer, FixedKLController, PPOConfig, RunningMoments
from transformers import pipeline
from transformers import AutoTokenizer
from trl import AutoModelForCausalLMWithValueHead
def _step_safety_checker(self, batch_size: int, queries: List[torch.LongTensor], responses: List[torch.LongTensor], scores: List[torch.FloatTensor], masks: Optional[List[torch.LongTensor]]=None):
    """
        Check if the input data is valid for training.

        Args:
            batch_size (int):
                Batch size from the config file.
            queries (List[`torch.LongTensor`]):
                List of tensors containing the encoded queries of shape (`query_length`)
            responses (List[`torch.LongTensor`]):
                List of tensors containing the encoded responses of shape (`response_length`)
            scores (List[`torch.FloatTensor`]):
                List of tensors containing the scores.
            masks (List[`torch.LongTensor`], *optional*):
                list of optional tensors containing the masks of shape (`query_length` + `response_length`)
        Returns:
            `tuple`: The input processed data.
        """
    for name, tensor_list in zip(['queries', 'responses', 'scores'], [queries, responses, scores]):
        if not isinstance(tensor_list, list):
            raise ValueError(f'{name} must be a list of tensors - got {type(tensor_list)}')
        if not isinstance(tensor_list[0], torch.Tensor):
            raise ValueError(f'Elements in {name} must be tensors - got {type(tensor_list[0])}')
        if batch_size is not None and len(tensor_list) != batch_size:
            raise ValueError(f'Batch size ({batch_size}) does not match number of examples - but got {len(tensor_list)} for: {name}')
    queries = [tensor.to(self.current_device) for tensor in queries]
    responses = [tensor.to(self.current_device) for tensor in responses]
    scores = [tensor.to(self.current_device) for tensor in scores]
    masks = [tensor.to(self.current_device) for tensor in masks] if masks is not None else None
    for i, score in enumerate(scores):
        if score.dim() > 1:
            raise ValueError(f'Scores must be 1-dimensional - got {score.dim()} for {score}')
        elif score.dim() == 1:
            scores[i] = score.squeeze()
    return (queries, responses, scores, masks)