import os
import re
import warnings
from shutil import copyfile
from typing import List, Optional, Tuple
from ...tokenization_utils_fast import PreTrainedTokenizerFast
from ...utils import is_sentencepiece_available, logging
@staticmethod
def _eventually_correct_t5_max_length(pretrained_model_name_or_path, max_model_length, init_max_model_length):
    if pretrained_model_name_or_path in T5TokenizerFast.max_model_input_sizes:
        deprecated_max_model_length = T5TokenizerFast.max_model_input_sizes[pretrained_model_name_or_path]
        if init_max_model_length is not None and init_max_model_length != max_model_length:
            return init_max_model_length
        elif init_max_model_length is None:
            warnings.warn(f'This tokenizer was incorrectly instantiated with a model max length of {deprecated_max_model_length} which will be corrected in Transformers v5.\nFor now, this behavior is kept to avoid breaking backwards compatibility when padding/encoding with `truncation is True`.\n- Be aware that you SHOULD NOT rely on {pretrained_model_name_or_path} automatically truncating your input to {deprecated_max_model_length} when padding/encoding.\n- If you want to encode/pad to sequences longer than {deprecated_max_model_length} you can either instantiate this tokenizer with `model_max_length` or pass `max_length` when encoding/padding.\n- To avoid this warning, please instantiate this tokenizer with `model_max_length` set to your preferred value.', FutureWarning)
    return max_model_length