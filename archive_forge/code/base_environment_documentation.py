import re
import warnings
from typing import Optional
import torch
from accelerate.utils import extract_model_from_parallel
from transformers import StoppingCriteria, StoppingCriteriaList
from ..import_utils import is_rich_available

        Generate responses for a list of query tensors.

        args:
            query_tensors (list[torch.Tensor]): A list of query tensors to generate responses for.
            batch_size (int): The batch size to use for generation.
            pad_to_multiple_of (int): The padding length to use for generation.
        