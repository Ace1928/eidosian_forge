from typing import Optional, Tuple
from ...tokenization_utils_fast import PreTrainedTokenizerFast
from ...utils import logging
from .tokenization_openai import OpenAIGPTTokenizer
class OpenAIGPTTokenizerFast(PreTrainedTokenizerFast):
    """
    Construct a "fast" GPT Tokenizer (backed by HuggingFace's *tokenizers* library). Based on Byte-Pair-Encoding with
    the following peculiarities:

    - lower case all inputs
    - uses BERT's BasicTokenizer for pre-BPE tokenization

    This tokenizer inherits from [`PreTrainedTokenizerFast`] which contains most of the main methods. Users should
    refer to this superclass for more information regarding those methods.

    Args:
        vocab_file (`str`):
            Path to the vocabulary file.
        merges_file (`str`):
            Path to the merges file.
        unk_token (`str`, *optional*, defaults to `"<unk>"`):
            The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
            token instead.
    """
    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    model_input_names = ['input_ids', 'attention_mask']
    slow_tokenizer_class = OpenAIGPTTokenizer

    def __init__(self, vocab_file=None, merges_file=None, tokenizer_file=None, unk_token='<unk>', **kwargs):
        super().__init__(vocab_file, merges_file, tokenizer_file=tokenizer_file, unk_token=unk_token, **kwargs)

    @property
    def do_lower_case(self):
        return True

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str]=None) -> Tuple[str]:
        files = self._tokenizer.model.save(save_directory, name=filename_prefix)
        return tuple(files)