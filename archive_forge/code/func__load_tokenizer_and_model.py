import math
import os
from collections import Counter, defaultdict
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Set, Tuple, Union
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from torchmetrics.utilities.data import _cumsum
from torchmetrics.utilities.imports import _TQDM_AVAILABLE, _TRANSFORMERS_GREATER_EQUAL_4_4
def _load_tokenizer_and_model(model_name_or_path: Union[str, os.PathLike], device: Optional[Union[str, torch.device]]=None) -> Tuple['PreTrainedTokenizerBase', 'PreTrainedModel']:
    """Load HuggingFace `transformers`' tokenizer and model. This function also handle a device placement.

    Args:
        model_name_or_path:
            A name or a model path used to load `transformers` pretrained model.
        device:
            A device to be used for calculation.

    Return:
        Initialized `transformers`' tokenizer and model.

    """
    from transformers import AutoModelForMaskedLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model = AutoModelForMaskedLM.from_pretrained(model_name_or_path)
    model.eval()
    model.to(device)
    return (tokenizer, model)