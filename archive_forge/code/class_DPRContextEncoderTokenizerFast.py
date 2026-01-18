import collections
from typing import List, Optional, Union
from ...tokenization_utils_base import BatchEncoding
from ...utils import TensorType, add_end_docstrings, add_start_docstrings, logging
from ..bert.tokenization_bert_fast import BertTokenizerFast
from .tokenization_dpr import DPRContextEncoderTokenizer, DPRQuestionEncoderTokenizer, DPRReaderTokenizer
class DPRContextEncoderTokenizerFast(BertTokenizerFast):
    """
    Construct a "fast" DPRContextEncoder tokenizer (backed by HuggingFace's *tokenizers* library).

    [`DPRContextEncoderTokenizerFast`] is identical to [`BertTokenizerFast`] and runs end-to-end tokenization:
    punctuation splitting and wordpiece.

    Refer to superclass [`BertTokenizerFast`] for usage examples and documentation concerning parameters.
    """
    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = CONTEXT_ENCODER_PRETRAINED_VOCAB_FILES_MAP
    max_model_input_sizes = CONTEXT_ENCODER_PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    pretrained_init_configuration = CONTEXT_ENCODER_PRETRAINED_INIT_CONFIGURATION
    slow_tokenizer_class = DPRContextEncoderTokenizer